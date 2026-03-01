use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use boxlite::BoxCommand;
use tokio::time::sleep;

use crate::boxlite_e2e::{BoxliteServices, BoxliteSshFixture, exec_box_checked};

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

    pub async fn read_remote_mcp_config(&self) -> Result<String> {
        exec_box_checked(
            &self.litebox,
            BoxCommand::new("cat").arg("/root/.kimi/mcp.json"),
        )
        .await
        .context("read remote MCP config")
    }

    pub fn local_mcp_config_path(&self) -> PathBuf {
        self.host_home().join(".kimi").join("mcp.json")
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
