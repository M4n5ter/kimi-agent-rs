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

        self.with_connection(move |conn| {
            let root_session_id = if let Some(parent_id) = parent_for_insert.as_ref() {
                let Some(root_id) = conn
                    .query_row(
                        "SELECT root_session_id FROM sessions WHERE id = ?1",
                        params![parent_id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
                else {
                    return Err(anyhow!("parent session not found: {parent_id}"));
                };
                let parent_workspace_id = conn.query_row(
                    "SELECT workspace_id FROM sessions WHERE id = ?1",
                    params![parent_id],
                    |row| row.get::<_, i64>(0),
                )?;
                if parent_workspace_id != workspace_id {
                    return Err(anyhow!("parent session must belong to the same workspace"));
                }
                root_id
            } else {
                session_id_for_insert.clone()
            };

            conn.execute(
                "
                INSERT INTO sessions (
                    id, workspace_id, parent_session_id, root_session_id, origin_kind, origin_json,
                    title, state, token_count, is_empty, next_context_seq, next_wire_seq,
                    created_at, updated_at, last_activity_at
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, 1, 0, 0, ?9, ?10, ?11)
                ",
                params![
                    session_id_for_insert,
                    workspace_id,
                    parent_for_insert,
                    root_session_id,
                    origin_kind,
                    origin_json,
                    title_for_insert,
                    state_text,
                    now,
                    now,
                    now,
                ],
            )?;
            Ok(())
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
                SELECT id, workspace_id, parent_session_id, root_session_id, origin_json, title,
                       state, token_count, is_empty, created_at, updated_at, last_activity_at
                FROM sessions
                WHERE workspace_id = ?1
                ORDER BY updated_at DESC, created_at DESC
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
        let Some(session_id) = workspace.last_active_session_id.clone() else {
            return Ok(None);
        };
        self.get_session(work_dir, kaos, &session_id).await
    }

    pub async fn finish_session(&self, finish: FinishSession) -> Result<()> {
        let now = now_epoch_secs();
        let state_text = finish.state.as_str().to_string();
        let session_id = finish.session_id.clone();
        let is_empty = finish.is_empty;
        self.with_connection(move |conn| {
            let tx = conn.transaction()?;

            let session_meta = tx
                .query_row(
                    "SELECT workspace_id, parent_session_id FROM sessions WHERE id = ?1",
                    params![session_id],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?)),
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
                params![finish.session_id, state_text, bool_to_i64(is_empty), now],
            )?;

            if parent_session_id.is_none() {
                if is_empty {
                    tx.execute(
                        "
                        UPDATE workspaces
                        SET last_active_session_id = NULL, updated_at = ?2
                        WHERE id = ?1 AND last_active_session_id = ?3
                        ",
                        params![workspace_id, now, finish.session_id],
                    )?;
                } else {
                    tx.execute(
                        "
                        UPDATE workspaces
                        SET last_active_session_id = ?2, updated_at = ?3
                        WHERE id = ?1
                        ",
                        params![workspace_id, finish.session_id, now],
                    )?;
                }
            }

            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn session_is_empty(&self, session_id: &str) -> Result<bool> {
        let session_id = session_id.to_string();
        self.with_connection(move |conn| {
            let has_events = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM session_events WHERE session_id = ?1)",
                params![session_id],
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
        SELECT id, workspace_id, parent_session_id, root_session_id, origin_json, title,
               state, token_count, is_empty, created_at, updated_at, last_activity_at
        FROM sessions
        WHERE id = ?1 AND workspace_id = ?2
        ",
        params![session_id, workspace_id],
        |row| session_from_row(row, kaos_scope_id.to_string(), work_dir.clone()),
    )
    .optional()
    .map_err(Into::into)
}

fn session_from_row(
    row: &rusqlite::Row<'_>,
    kaos_scope_id: String,
    work_dir: KaosPath,
) -> rusqlite::Result<SessionRecord> {
    let origin_json: String = row.get(4)?;
    let origin = serde_json::from_str::<SessionOrigin>(&origin_json).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(err))
    })?;
    let state_text: String = row.get(6)?;
    let state = SessionState::parse(&state_text).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(
            6,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                err.to_string(),
            )),
        )
    })?;

    Ok(SessionRecord {
        id: row.get(0)?,
        workspace_id: row.get(1)?,
        kaos_scope_id,
        work_dir,
        parent_session_id: row.get(2)?,
        root_session_id: row.get(3)?,
        origin,
        title: row.get(5)?,
        state,
        token_count: row.get(7)?,
        is_empty: row.get::<_, i64>(8)? != 0,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
        last_activity_at: row.get(11)?,
    })
}
