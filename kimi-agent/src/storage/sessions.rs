use anyhow::{Context, Result, anyhow};
use kaos::KaosPath;
use rusqlite::{Connection, OptionalExtension, params};

use super::db::{bool_to_i64, now_epoch_secs};
use super::{CreateSession, FinishSession, SessionOrigin, SessionRecord, SessionState, Storage};
use crate::config::KaosConfig;
use crate::session_id::normalize_session_id;

impl Storage {
    pub async fn create_session(&self, input: CreateSession) -> Result<SessionRecord> {
        let session_id = input
            .session_id
            .map(|session_id| normalize_session_id(&session_id))
            .transpose()?
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let workspace = self.resolve_workspace(&input.work_dir, &input.kaos).await?;
        let title = input
            .title
            .unwrap_or_else(|| format!("Untitled ({session_id})"));
        let state = input.state;
        let origin = input.origin;
        let parent_session_id = input.parent_session_id;
        let workspace_id = workspace.id;
        let kaos_scope_id = workspace.kaos_scope_id.clone();
        let work_dir = KaosPath::new(&workspace.canonical_path);
        let now = now_epoch_secs();
        let title_for_insert = title.clone();
        let session_id_for_insert = session_id.clone();
        let origin_json = serde_json::to_string(&origin).context("serialize session origin")?;
        let origin_kind = origin.kind_str().to_string();
        let state_text = state.as_str().to_string();
        let session_id_for_query = session_id.clone();
        let parent_for_insert = parent_session_id.clone();

        let inserted_id = self
            .with_connection(move |conn| {
                let tx = conn.transaction()?;
                let session_db_id = allocate_session_id(&tx)?;
                let (parent_db_id, root_session_db_id) =
                    if let Some(parent_id) = parent_for_insert.as_ref() {
                        let Some((parent_db_id, root_db_id)) = tx
                            .query_row(
                                "
                                SELECT id, root_session_id
                                FROM sessions
                                WHERE workspace_id = ?1 AND session_id = ?2
                                ",
                                params![workspace_id, parent_id],
                                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                            )
                            .optional()?
                        else {
                            return Err(anyhow!("parent session not found: {parent_id}"));
                        };
                        (Some(parent_db_id), root_db_id)
                    } else {
                        (None, session_db_id)
                    };

                tx.execute(
                    "
                    INSERT INTO sessions (
                        id, workspace_id, session_id, parent_session_id, root_session_id,
                        origin_kind, origin_json, title, state, token_count, is_empty,
                        next_context_seq, next_wire_seq, created_at, updated_at, last_activity_at
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, 1, 0, 0, ?10, ?11, ?12)
                    ",
                    params![
                        session_db_id,
                        workspace_id,
                        session_id_for_insert,
                        parent_db_id,
                        root_session_db_id,
                        origin_kind,
                        origin_json,
                        title_for_insert,
                        state_text,
                        now,
                        now,
                        now,
                    ],
                )?;
                tx.commit()?;
                Ok(session_db_id)
            })
            .await?;

        self.with_connection(move |conn| {
            load_session(
                conn,
                &session_id_for_query,
                workspace_id,
                &kaos_scope_id,
                &work_dir,
            )?
            .ok_or_else(|| anyhow!("session disappeared after insert"))
            .and_then(|record| {
                if record.db_id == inserted_id {
                    Ok(record)
                } else {
                    Err(anyhow!("loaded unexpected session after insert"))
                }
            })
        })
        .await
    }

    pub async fn get_session(
        &self,
        work_dir: &KaosPath,
        kaos: &KaosConfig,
        session_id: &str,
    ) -> Result<Option<SessionRecord>> {
        let session_id = normalize_session_id(session_id)?;
        let workspace = self.resolve_workspace(work_dir, kaos).await?;
        let kaos_scope_id = workspace.kaos_scope_id.clone();
        let workspace_id = workspace.id;
        let work_dir = KaosPath::new(&workspace.canonical_path);
        self.with_connection(move |conn| {
            load_session(conn, &session_id, workspace_id, &kaos_scope_id, &work_dir)
        })
        .await
    }

    pub async fn list_sessions(
        &self,
        work_dir: &KaosPath,
        kaos: &KaosConfig,
    ) -> Result<Vec<SessionRecord>> {
        let workspace = self.resolve_workspace(work_dir, kaos).await?;
        let workspace_id = workspace.id;
        let kaos_scope_id = workspace.kaos_scope_id.clone();
        let work_dir = KaosPath::new(&workspace.canonical_path);
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "
                SELECT s.id, s.session_id, s.workspace_id, parent.session_id, root.session_id,
                       s.origin_json, s.title, s.state, s.token_count, s.is_empty,
                       s.created_at, s.updated_at, s.last_activity_at
                FROM sessions AS s
                LEFT JOIN sessions AS parent ON parent.id = s.parent_session_id
                JOIN sessions AS root ON root.id = s.root_session_id
                WHERE s.workspace_id = ?1
                ORDER BY s.updated_at DESC, s.created_at DESC
                ",
            )?;
            let rows = stmt.query_map(params![workspace_id], |row| {
                session_from_row(row, kaos_scope_id.clone(), work_dir.clone())
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
        .await
    }

    pub async fn continue_session(
        &self,
        work_dir: &KaosPath,
        kaos: &KaosConfig,
    ) -> Result<Option<SessionRecord>> {
        let workspace = self.resolve_workspace(work_dir, kaos).await?;
        let Some(session_db_id) = workspace.last_active_session_id else {
            return Ok(None);
        };
        let kaos_scope_id = workspace.kaos_scope_id.clone();
        let workspace_id = workspace.id;
        let work_dir = KaosPath::new(&workspace.canonical_path);
        self.with_connection(move |conn| {
            load_session_by_db_id(conn, session_db_id, workspace_id, &kaos_scope_id, &work_dir)
        })
        .await
    }

    pub async fn finish_session(&self, finish: FinishSession) -> Result<()> {
        let now = now_epoch_secs();
        let state_text = finish.state.as_str().to_string();
        let session_db_id = finish.session_db_id;
        let is_empty = finish.is_empty;
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;

            let session_meta = tx
                .query_row(
                    "SELECT workspace_id, parent_session_id FROM sessions WHERE id = ?1",
                    params![session_db_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?)),
                )
                .optional()?;
            let Some((workspace_id, parent_session_id)) = session_meta else {
                return Ok(());
            };

            tx.execute(
                "
                UPDATE sessions
                SET state = ?2, is_empty = ?3, updated_at = ?4, last_activity_at = ?4
                WHERE id = ?1
                ",
                params![session_db_id, state_text, bool_to_i64(is_empty), now],
            )?;

            if parent_session_id.is_none() {
                if is_empty {
                    tx.execute(
                        "
                        UPDATE workspaces
                        SET last_active_session_id = NULL, updated_at = ?2
                        WHERE id = ?1 AND last_active_session_id = ?3
                        ",
                        params![workspace_id, now, session_db_id],
                    )?;
                } else {
                    tx.execute(
                        "
                        UPDATE workspaces
                        SET last_active_session_id = ?2, updated_at = ?3
                        WHERE id = ?1
                        ",
                        params![workspace_id, session_db_id, now],
                    )?;
                }
            }

            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn session_is_empty(&self, session_db_id: i64) -> Result<bool> {
        self.with_connection(move |conn| {
            let has_events = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM session_events WHERE session_id = ?1)",
                params![session_db_id],
                |row| row.get::<_, i64>(0),
            )?;
            Ok(has_events == 0)
        })
        .await
    }
}

pub(super) fn load_session(
    conn: &Connection,
    session_id: &str,
    workspace_id: i64,
    kaos_scope_id: &str,
    work_dir: &KaosPath,
) -> Result<Option<SessionRecord>> {
    conn.query_row(
        "
        SELECT s.id, s.session_id, s.workspace_id, parent.session_id, root.session_id,
               s.origin_json, s.title, s.state, s.token_count, s.is_empty,
               s.created_at, s.updated_at, s.last_activity_at
        FROM sessions AS s
        LEFT JOIN sessions AS parent ON parent.id = s.parent_session_id
        JOIN sessions AS root ON root.id = s.root_session_id
        WHERE s.session_id = ?1 AND s.workspace_id = ?2
        ",
        params![session_id, workspace_id],
        |row| session_from_row(row, kaos_scope_id.to_string(), work_dir.clone()),
    )
    .optional()
    .map_err(Into::into)
}

fn load_session_by_db_id(
    conn: &Connection,
    session_db_id: i64,
    workspace_id: i64,
    kaos_scope_id: &str,
    work_dir: &KaosPath,
) -> Result<Option<SessionRecord>> {
    conn.query_row(
        "
        SELECT s.id, s.session_id, s.workspace_id, parent.session_id, root.session_id,
               s.origin_json, s.title, s.state, s.token_count, s.is_empty,
               s.created_at, s.updated_at, s.last_activity_at
        FROM sessions AS s
        LEFT JOIN sessions AS parent ON parent.id = s.parent_session_id
        JOIN sessions AS root ON root.id = s.root_session_id
        WHERE s.id = ?1 AND s.workspace_id = ?2
        ",
        params![session_db_id, workspace_id],
        |row| session_from_row(row, kaos_scope_id.to_string(), work_dir.clone()),
    )
    .optional()
    .map_err(Into::into)
}

fn allocate_session_id(conn: &Connection) -> Result<i64> {
    let next_id = conn.query_row("SELECT COALESCE(MAX(id), 0) + 1 FROM sessions", [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(next_id)
}

fn session_from_row(
    row: &rusqlite::Row<'_>,
    kaos_scope_id: String,
    work_dir: KaosPath,
) -> rusqlite::Result<SessionRecord> {
    let origin_json: String = row.get(5)?;
    let origin = serde_json::from_str::<SessionOrigin>(&origin_json).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(err))
    })?;
    let state_text: String = row.get(7)?;
    let state = SessionState::parse(&state_text).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(
            7,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                err.to_string(),
            )),
        )
    })?;

    Ok(SessionRecord {
        db_id: row.get(0)?,
        id: row.get(1)?,
        workspace_id: row.get(2)?,
        kaos_scope_id,
        work_dir,
        parent_session_id: row.get(3)?,
        root_session_id: row.get(4)?,
        origin,
        title: row.get(6)?,
        state,
        token_count: row.get(8)?,
        is_empty: row.get::<_, i64>(9)? != 0,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
        last_activity_at: row.get(12)?,
    })
}
