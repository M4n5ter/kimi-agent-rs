use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use boxlite::BoxCommand;
use serde_json::{Map, Value, json};
use tokio::time::sleep;

use crate::boxlite_e2e::{BoxliteServices, BoxliteSshFixture};

pub use crate::boxlite_e2e::{GUEST_FIXTURE_DIR, GUEST_PYTHON, HTTP_ENV_VALUE};

pub const STDIO_SERVER_NAME: &str = "box-stdio";
pub const HTTP_SERVER_NAME: &str = "box-http";
pub const STDIO_TOOL_NAME: &str = "box_stdio_context";
pub const HTTP_TOOL_NAME: &str = "box_http_context";
pub const GUEST_STDIO_SCRIPT: &str = "/root/fixtures/boxlite_mcp_stdio.py";
pub const STDIO_ENV_VALUE: &str = "runtime-stdio";

pub async fn provision_mcp_fixture() -> Result<BoxliteSshFixture> {
    BoxliteSshFixture::provision(BoxliteServices {
        http: true,
        oauth: false,
    })
    .await
}

impl BoxliteSshFixture {
    pub fn guest_http_port(&self) -> u16 {
        self.http
            .as_ref()
            .expect("HTTP service must be provisioned for MCP BoxLite tests")
            .guest_port
    }

    pub fn local_legacy_mcp_config_path(&self) -> PathBuf {
        self.host_home().join(".kimi").join("mcp.json")
    }

    pub async fn remote_legacy_mcp_config_exists(&self) -> Result<bool> {
        self.remote_file_exists("/root/.kimi/mcp.json").await
    }

    pub fn read_host_mcp_config(&self) -> Result<Value> {
        self.with_local_state_db(|conn| {
            let rows = {
                let mut stmt = conn.prepare(
                    "
                    SELECT mcp_servers.name, mcp_servers.config_json
                    FROM mcp_servers
                    JOIN kaos_scopes ON kaos_scopes.id = mcp_servers.kaos_scope_id
                    WHERE kaos_scopes.kind = 'ssh'
                    ORDER BY mcp_servers.name ASC
                    ",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };

            let mut servers = Map::new();
            for (name, config_json) in rows {
                let config: Value =
                    serde_json::from_str(&config_json).context("parse MCP server config JSON")?;
                servers.insert(name, config);
            }

            Ok(json!({ "mcpServers": servers }))
        })
    }

    pub async fn wait_for_remote_process_exit(&self, pid: u32) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if !self.remote_process_exists(pid).await? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!("remote process {pid} is still alive after cleanup");
            }
            sleep(Duration::from_millis(250)).await;
        }
    }

    pub async fn remote_process_exists(&self, pid: u32) -> Result<bool> {
        let mut execution = self
            .litebox
            .exec(BoxCommand::new("sh").args(["-lc", &format!("kill -0 {pid}")]))
            .await
            .context("probe remote process existence")?;
        let result = execution.wait().await.context("wait for process probe")?;
        Ok(result.exit_code == 0)
    }
}
