use anyhow::{Context, Result};
use rmcp::transport::auth::StoredCredentials;
use rusqlite::{OptionalExtension, params};
use serde_json::Value;

use super::db::now_epoch_secs;
use super::{McpServerRecord, Storage};
use crate::config::KaosConfig;

impl Storage {
    pub async fn load_mcp_servers(&self, kaos: &KaosConfig) -> Result<Vec<McpServerRecord>> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "
                SELECT name, transport_kind, config_json
                FROM mcp_servers
                WHERE kaos_scope_id = ?1
                ORDER BY name ASC
                ",
            )?;
            let rows = stmt.query_map(params![scope_id], |row| {
                let name: String = row.get(0)?;
                let transport_kind: String = row.get(1)?;
                let config_json: String = row.get(2)?;
                let config = serde_json::from_str::<Value>(&config_json).map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(err),
                    )
                })?;
                Ok(McpServerRecord {
                    name,
                    transport_kind,
                    config,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(Into::into)
        })
        .await
    }

    pub async fn upsert_mcp_server(
        &self,
        kaos: &KaosConfig,
        name: &str,
        config: &Value,
    ) -> Result<()> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        let name = name.to_string();
        let config = config.clone();
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let transport_kind = infer_mcp_transport_kind(&config);
            let config_json = serde_json::to_string(&config).context("serialize MCP server config")?;
            conn.execute(
                "
                INSERT INTO mcp_servers (kaos_scope_id, name, transport_kind, config_json, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(kaos_scope_id, name) DO UPDATE SET
                    transport_kind = excluded.transport_kind,
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                ",
                params![scope_id, name, transport_kind, config_json, now, now],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn delete_mcp_server(&self, kaos: &KaosConfig, name: &str) -> Result<bool> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        let name = name.to_string();
        self.with_connection(move |conn| {
            let deleted = conn.execute(
                "DELETE FROM mcp_servers WHERE kaos_scope_id = ?1 AND name = ?2",
                params![scope_id, name],
            )?;
            Ok(deleted > 0)
        })
        .await
    }

    pub async fn load_mcp_credentials(
        &self,
        kaos: &KaosConfig,
        server_url: &str,
    ) -> Result<Option<StoredCredentials>> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        let server_url = server_url.to_string();
        self.with_connection(move |conn| {
            let credentials_json = conn
                .query_row(
                    "
                    SELECT credentials_json
                    FROM mcp_credentials
                    WHERE kaos_scope_id = ?1 AND server_url = ?2
                    ",
                    params![scope_id, server_url],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            credentials_json
                .map(|text| {
                    serde_json::from_str::<StoredCredentials>(&text)
                        .context("deserialize MCP credentials")
                })
                .transpose()
        })
        .await
    }

    pub async fn save_mcp_credentials(
        &self,
        kaos: &KaosConfig,
        server_url: &str,
        credentials: &StoredCredentials,
    ) -> Result<()> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        let server_url = server_url.to_string();
        let credentials = credentials.clone();
        let now = now_epoch_secs();
        self.with_connection(move |conn| {
            let credentials_json =
                serde_json::to_string(&credentials).context("serialize MCP credentials")?;
            conn.execute(
                "
                INSERT INTO mcp_credentials (kaos_scope_id, server_url, credentials_json, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5)
                ON CONFLICT(kaos_scope_id, server_url) DO UPDATE SET
                    credentials_json = excluded.credentials_json,
                    updated_at = excluded.updated_at
                ",
                params![scope_id, server_url, credentials_json, now, now],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn clear_mcp_credentials(&self, kaos: &KaosConfig, server_url: &str) -> Result<()> {
        let scope = self.ensure_kaos_scope(kaos).await?;
        let scope_id = scope.id;
        let server_url = server_url.to_string();
        self.with_connection(move |conn| {
            conn.execute(
                "DELETE FROM mcp_credentials WHERE kaos_scope_id = ?1 AND server_url = ?2",
                params![scope_id, server_url],
            )?;
            Ok(())
        })
        .await
    }
}

fn infer_mcp_transport_kind(config: &Value) -> &'static str {
    if config.get("command").and_then(Value::as_str).is_some() {
        "stdio"
    } else {
        "http"
    }
}
