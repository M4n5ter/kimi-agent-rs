use anyhow::{Result, anyhow};
use kaos::KaosPath;
use rusqlite::{Connection, OptionalExtension, params};

use super::db::now_epoch_secs;
use super::{KaosScopeKind, KaosScopeRecord, Storage, WorkspaceRecord};
use crate::config::KaosConfig;

impl Storage {
    pub async fn ensure_kaos_scope(&self, kaos: &KaosConfig) -> Result<KaosScopeRecord> {
        let record = kaos_scope_record(kaos)?;
        let upsert = record.clone();
        self.with_connection(move |conn| {
            conn.execute(
                "
                INSERT INTO kaos_scopes (id, kind, display_name, definition_json, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    display_name = excluded.display_name,
                    definition_json = excluded.definition_json,
                    updated_at = excluded.updated_at
                ",
                params![
                    upsert.id,
                    upsert.kind.as_str(),
                    upsert.display_name,
                    upsert.definition_json,
                    upsert.created_at,
                    upsert.updated_at,
                ],
            )?;
            Ok(())
        })
        .await?;
        Ok(record)
    }

    pub async fn resolve_workspace(
        &self,
        work_dir: &KaosPath,
        kaos: &KaosConfig,
    ) -> Result<WorkspaceRecord> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let canonical_path = work_dir.canonical().to_string_lossy();
        let display_path = work_dir.to_string_lossy();
        let scope_id = scope.id.clone();
        let inserted_scope_id = scope_id.clone();
        let inserted_canonical_path = canonical_path.clone();
        let inserted_display_path = display_path.clone();
        let now = now_epoch_secs();

        self.with_connection(move |conn| {
            conn.execute(
                "
                INSERT INTO workspaces (kaos_scope_id, canonical_path, display_path, last_active_session_id, created_at, updated_at)
                VALUES (?1, ?2, ?3, NULL, ?4, ?5)
                ON CONFLICT(kaos_scope_id, canonical_path) DO UPDATE SET
                    display_path = excluded.display_path,
                    updated_at = excluded.updated_at
                ",
                params![
                    inserted_scope_id,
                    inserted_canonical_path,
                    inserted_display_path,
                    now,
                    now,
                ],
            )?;
            Ok(())
        })
        .await?;

        self.with_connection(move |conn| {
            load_workspace(conn, &scope_id, &canonical_path)?
                .ok_or_else(|| anyhow!("workspace disappeared after upsert"))
        })
        .await
    }
}

pub(super) fn load_workspace(
    conn: &Connection,
    scope_id: &str,
    canonical_path: &str,
) -> Result<Option<WorkspaceRecord>> {
    conn.query_row(
        "
        SELECT id, kaos_scope_id, canonical_path, display_path, last_active_session_id, created_at, updated_at
        FROM workspaces
        WHERE kaos_scope_id = ?1 AND canonical_path = ?2
        ",
        params![scope_id, canonical_path],
        |row| {
            Ok(WorkspaceRecord {
                id: row.get(0)?,
                kaos_scope_id: row.get(1)?,
                canonical_path: row.get(2)?,
                display_path: row.get(3)?,
                last_active_session_id: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn kaos_scope_record(kaos: &KaosConfig) -> Result<KaosScopeRecord> {
    let now = now_epoch_secs();
    match kaos {
        KaosConfig::Local => Ok(KaosScopeRecord {
            id: "local".to_string(),
            kind: KaosScopeKind::Local,
            display_name: "local".to_string(),
            definition_json: serde_json::json!({ "type": "local" }).to_string(),
            created_at: now,
            updated_at: now,
        }),
        KaosConfig::Ssh { options } => {
            let username = options
                .username
                .clone()
                .unwrap_or_else(default_ssh_username);
            Ok(KaosScopeRecord {
                id: options.logical_storage_name(),
                kind: KaosScopeKind::Ssh,
                display_name: format!("{username}@{}:{}", options.host, options.port),
                definition_json: serde_json::json!({
                    "type": "ssh",
                    "host": options.host,
                    "port": options.port,
                    "username": options.username,
                    "cwd": options.cwd,
                    "host_key_policy": options.host_key_policy,
                })
                .to_string(),
                created_at: now,
                updated_at: now,
            })
        }
    }
}

fn default_ssh_username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "root".to_string())
}
