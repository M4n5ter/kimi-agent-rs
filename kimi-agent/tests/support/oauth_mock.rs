use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use boxlite::BoxCommand;
use rusqlite::OptionalExtension;
use serde::Deserialize;

use crate::boxlite_e2e::{
    BoxliteServices, BoxliteSshFixture, GUEST_OAUTH_STATE_FILE, exec_box_checked,
};

const OAUTH_SERVER_NAME: &str = "box-oauth-http";
const OAUTH_TOOL_NAME: &str = "box_oauth_http_context";
const OAUTH_BROWSER_SCRIPT: &str = include_str!("../fixtures/boxlite_e2e/oauth_browser.sh");
const ACCESS_TOKEN: &str = "boxlite-oauth-access-token";
const REMOTE_MCP_AUTH_FILE: &str = "/root/.kimi/credentials/mcp_auth.json";

#[derive(Debug, Deserialize)]
pub struct OAuthFixtureState {
    pub last_authorization_header: Option<String>,
    pub authorized_mcp_requests: u64,
    pub register_requests: u64,
    pub authorize_requests: u64,
    pub token_requests: u64,
    pub resource_metadata_requests: u64,
    pub authorization_metadata_requests: u64,
}

pub struct HostOauthFixture {
    remote_mcp_url: String,
    remote_authority: String,
    local_authority: String,
}

pub async fn provision_oauth_fixture() -> Result<BoxliteSshFixture> {
    BoxliteSshFixture::provision(BoxliteServices {
        http: false,
        oauth: true,
    })
    .await
}

impl HostOauthFixture {
    pub async fn start(fixture: &BoxliteSshFixture) -> Result<Self> {
        let oauth = fixture
            .oauth
            .as_ref()
            .context("OAuth service must be provisioned for OAuth BoxLite tests")?;
        Ok(Self {
            remote_mcp_url: format!("http://127.0.0.1:{}/mcp", oauth.guest_port),
            remote_authority: format!("127.0.0.1:{}", oauth.guest_port),
            local_authority: format!(
                "127.0.0.1:{}",
                oauth
                    .host_forward_port
                    .context("missing OAuth host forward port")?
            ),
        })
    }

    pub fn server_name(&self) -> &'static str {
        OAUTH_SERVER_NAME
    }

    pub fn tool_name(&self) -> &'static str {
        OAUTH_TOOL_NAME
    }

    pub fn remote_mcp_url(&self) -> String {
        self.remote_mcp_url.clone()
    }

    pub fn remote_authority(&self) -> &str {
        &self.remote_authority
    }

    pub fn local_authority(&self) -> &str {
        &self.local_authority
    }

    pub async fn shutdown(self) -> Result<()> {
        Ok(())
    }

    pub async fn write_browser_script(&self, host_home: &Path) -> Result<PathBuf> {
        let path = host_home.join("oauth-browser.sh");
        std::fs::write(&path, OAUTH_BROWSER_SCRIPT)
            .with_context(|| format!("write {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = std::fs::metadata(&path)
                .with_context(|| format!("stat {}", path.display()))?
                .permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&path, permissions)
                .with_context(|| format!("chmod {}", path.display()))?;
        }
        Ok(path)
    }

    pub async fn last_authorization_header(
        &self,
        fixture: &BoxliteSshFixture,
    ) -> Result<Option<String>> {
        Ok(self.read_state(fixture).await?.last_authorization_header)
    }

    pub async fn read_state(&self, fixture: &BoxliteSshFixture) -> Result<OAuthFixtureState> {
        let text = fixture.read_guest_oauth_state().await?;
        serde_json::from_str(&text).context("parse guest OAuth state file")
    }

    pub async fn dump_debug_artifacts(&self, fixture: &BoxliteSshFixture) {
        match fixture.read_guest_oauth_state().await {
            Ok(text) => eprintln!("--- guest oauth state ({GUEST_OAUTH_STATE_FILE}) ---\n{text}"),
            Err(err) => eprintln!("--- guest oauth state unavailable ---\n{err:#}"),
        }
    }
}

pub fn assert_sqlite_auth_record_contains_token(text: &str) -> Result<()> {
    let data: serde_json::Value =
        serde_json::from_str(text).context("parse SQLite-backed OAuth credential record")?;
    let access_token = data["token_response"]["access_token"]
        .as_str()
        .context("missing access token in OAuth credential record")?;
    if access_token != ACCESS_TOKEN {
        bail!("unexpected OAuth access token: {access_token}");
    }
    Ok(())
}

impl BoxliteSshFixture {
    pub fn local_legacy_mcp_auth_path(&self) -> PathBuf {
        self.host_home()
            .join(".kimi")
            .join("credentials")
            .join("mcp_auth.json")
    }

    pub async fn remote_legacy_mcp_auth_exists(&self) -> Result<bool> {
        self.remote_file_exists(REMOTE_MCP_AUTH_FILE).await
    }

    pub fn read_host_mcp_credential(&self, server_url: &str) -> Result<Option<String>> {
        let server_url = server_url.to_string();
        self.with_local_state_db(move |conn| {
            conn.query_row(
                "
                SELECT mcp_credentials.credentials_json
                FROM mcp_credentials
                JOIN kaos_scopes ON kaos_scopes.id = mcp_credentials.kaos_scope_id
                WHERE kaos_scopes.kind = 'ssh' AND mcp_credentials.server_url = ?1
                ",
                [server_url],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub async fn read_guest_oauth_state(&self) -> Result<String> {
        exec_box_checked(
            &self.litebox,
            BoxCommand::new("cat").arg(GUEST_OAUTH_STATE_FILE),
        )
        .await
        .context("read guest OAuth state file")
    }
}
