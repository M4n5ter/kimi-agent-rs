use std::collections::HashMap;

use kaos::{KaosPath, get_current_kaos, is_not_found_error};
use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::exception::MCPConfigError;

async fn get_mcp_state_dir() -> KaosPath {
    match kaos::env_var("KIMI_SHARE_DIR").await {
        Ok(Some(path)) if !path.trim().is_empty() => KaosPath::new(&path),
        _ => get_current_kaos().app_state_dir("kimi"),
    }
}

pub async fn get_global_mcp_config_file() -> KaosPath {
    get_mcp_state_dir().await / "mcp.json"
}

pub async fn load_mcp_config_file(path: &KaosPath) -> Result<Value, MCPConfigError> {
    let text = path
        .read_text()
        .await
        .map_err(|err| MCPConfigError::new(format!("Failed to read MCP config file: {err}")))?;
    load_mcp_config_string(&text)
}

pub fn load_mcp_config_string(text: &str) -> Result<Value, MCPConfigError> {
    let mut value: Value = serde_json::from_str(text)
        .map_err(|err| MCPConfigError::new(format!("Invalid JSON: {err}")))?;
    ensure_mcp_servers(&mut value)?;
    Ok(value)
}

pub fn ensure_mcp_servers(value: &mut Value) -> Result<&mut Map<String, Value>, MCPConfigError> {
    let map = value
        .as_object_mut()
        .ok_or_else(|| MCPConfigError::new("MCP config must be a JSON object"))?;
    if !map.contains_key("mcpServers") {
        map.insert("mcpServers".to_string(), json!({}));
    }
    let servers = map
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| MCPConfigError::new("mcpServers must be a JSON object"))?;
    Ok(servers)
}

pub async fn save_mcp_config(value: &Value) -> Result<(), MCPConfigError> {
    let path = get_global_mcp_config_file().await;
    path.parent()
        .mkdir(true, true)
        .await
        .map_err(|err| MCPConfigError::new(format!("Failed to create MCP config dir: {err}")))?;
    let text = serde_json::to_string_pretty(value)
        .map_err(|err| MCPConfigError::new(format!("Failed to serialize MCP config: {err}")))?;
    path.write_text(&text)
        .await
        .map_err(|err| MCPConfigError::new(format!("Failed to write MCP config file: {err}")))?;
    Ok(())
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct McpCredentialFile {
    #[serde(default)]
    servers: HashMap<String, StoredCredentials>,
}

#[derive(Debug, Clone)]
pub struct KaosCredentialStore {
    path: KaosPath,
    server_key: String,
}

impl KaosCredentialStore {
    pub fn new(path: KaosPath, server_key: impl Into<String>) -> Self {
        Self {
            path,
            server_key: server_key.into(),
        }
    }

    async fn read_credential_file(&self) -> Result<Option<McpCredentialFile>, AuthError> {
        let text = match self.path.read_text().await {
            Ok(text) => text,
            Err(err) if is_not_found_error(&err) => return Ok(None),
            Err(err) => {
                return Err(AuthError::InternalError(format!(
                    "Failed to read MCP auth file: {err}"
                )));
            }
        };
        let data = serde_json::from_str::<McpCredentialFile>(&text)
            .map_err(|err| AuthError::InternalError(format!("Invalid MCP auth file: {err}")))?;
        Ok(Some(data))
    }
}

#[async_trait::async_trait]
impl CredentialStore for KaosCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let Some(data) = self.read_credential_file().await? else {
            return Ok(None);
        };
        Ok(data.servers.get(&self.server_key).cloned())
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let mut data = self.read_credential_file().await?.unwrap_or_default();
        data.servers.insert(self.server_key.clone(), credentials);
        self.path.parent().mkdir(true, true).await.map_err(|err| {
            AuthError::InternalError(format!("Failed to create MCP auth dir: {err}"))
        })?;
        let text = serde_json::to_string_pretty(&data).map_err(|err| {
            AuthError::InternalError(format!("Failed to serialize MCP auth file: {err}"))
        })?;
        self.path.write_text(&text).await.map_err(|err| {
            AuthError::InternalError(format!("Failed to write MCP auth file: {err}"))
        })?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let mut data = self.read_credential_file().await?.unwrap_or_default();
        data.servers.remove(&self.server_key);
        self.path.parent().mkdir(true, true).await.map_err(|err| {
            AuthError::InternalError(format!("Failed to create MCP auth dir: {err}"))
        })?;
        let text = serde_json::to_string_pretty(&data).map_err(|err| {
            AuthError::InternalError(format!("Failed to serialize MCP auth file: {err}"))
        })?;
        self.path.write_text(&text).await.map_err(|err| {
            AuthError::InternalError(format!("Failed to write MCP auth file: {err}"))
        })?;
        Ok(())
    }
}

pub async fn get_mcp_auth_file() -> KaosPath {
    get_mcp_state_dir().await / "credentials" / "mcp_auth.json"
}

pub async fn get_mcp_credential_store(server_url: &str) -> KaosCredentialStore {
    KaosCredentialStore::new(get_mcp_auth_file().await, server_url)
}

pub async fn has_oauth_tokens(server_url: &str) -> Result<bool, AuthError> {
    let store = get_mcp_credential_store(server_url).await;
    Ok(store
        .load()
        .await?
        .and_then(|creds| creds.token_response)
        .is_some())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use anyhow::Result;
    use kaos::{
        AsyncReadWrite, ExecOptions, Kaos, KaosFileError, KaosFileErrorKind, KaosPath,
        KaosPlatform, KaosProcess, LineStream, LocalKaos, StrOrKaosPath, reset_current_kaos,
        set_current_kaos, with_current_kaos_scope,
    };
    use rmcp::transport::auth::{CredentialStore, OAuthTokenResponse, StoredCredentials};
    use serde_json::json;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::{
        KaosCredentialStore, get_global_mcp_config_file, get_mcp_credential_store,
        has_oauth_tokens, load_mcp_config_file, save_mcp_config,
    };

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::const_new(()));

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: tests serialize env mutations via ENV_LOCK.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(prev) => {
                    // SAFETY: tests serialize env mutations via ENV_LOCK.
                    unsafe {
                        std::env::set_var(self.key, prev);
                    }
                }
                None => {
                    // SAFETY: tests serialize env mutations via ENV_LOCK.
                    unsafe {
                        std::env::remove_var(self.key);
                    }
                }
            }
        }
    }

    struct CredentialReadFailKaos {
        inner: LocalKaos,
        auth_path: KaosPath,
    }

    impl CredentialReadFailKaos {
        fn new(auth_path: KaosPath) -> Self {
            Self {
                inner: LocalKaos::new(),
                auth_path,
            }
        }
    }

    #[async_trait::async_trait]
    impl Kaos for CredentialReadFailKaos {
        fn name(&self) -> &str {
            "credential-read-fail"
        }

        fn platform(&self) -> KaosPlatform {
            self.inner.platform()
        }

        fn normpath(&self, path: &StrOrKaosPath<'_>) -> KaosPath {
            self.inner.normpath(path)
        }

        fn home(&self) -> KaosPath {
            self.inner.home()
        }

        fn cwd(&self) -> KaosPath {
            self.inner.cwd()
        }

        async fn chdir(&self, path: &KaosPath) -> Result<()> {
            self.inner.chdir(path).await
        }

        async fn stat(&self, path: &KaosPath, follow_symlinks: bool) -> Result<kaos::StatResult> {
            self.inner.stat(path, follow_symlinks).await
        }

        async fn iterdir(&self, path: &KaosPath) -> Result<Vec<KaosPath>> {
            self.inner.iterdir(path).await
        }

        async fn glob(
            &self,
            path: &KaosPath,
            pattern: &str,
            case_sensitive: bool,
        ) -> Result<Vec<KaosPath>> {
            self.inner.glob(path, pattern, case_sensitive).await
        }

        async fn read_bytes(&self, path: &KaosPath, limit: Option<usize>) -> Result<Vec<u8>> {
            self.inner.read_bytes(path, limit).await
        }

        async fn read_text(&self, path: &KaosPath) -> Result<String> {
            if *path == self.auth_path {
                return Err(KaosFileError::new(
                    path,
                    "read text",
                    KaosFileErrorKind::PermissionDenied,
                    "simulated permission denied",
                )
                .into());
            }
            self.inner.read_text(path).await
        }

        async fn read_lines(&self, path: &KaosPath) -> Result<Vec<String>> {
            self.inner.read_lines(path).await
        }

        async fn read_lines_stream(&self, path: &KaosPath) -> Result<LineStream> {
            self.inner.read_lines_stream(path).await
        }

        async fn write_bytes(&self, path: &KaosPath, data: &[u8]) -> Result<usize> {
            self.inner.write_bytes(path, data).await
        }

        async fn write_text(&self, path: &KaosPath, data: &str, append: bool) -> Result<usize> {
            self.inner.write_text(path, data, append).await
        }

        async fn chmod(&self, path: &KaosPath, mode: u32) -> Result<()> {
            self.inner.chmod(path, mode).await
        }

        async fn mkdir(&self, path: &KaosPath, parents: bool, exist_ok: bool) -> Result<()> {
            self.inner.mkdir(path, parents, exist_ok).await
        }

        async fn env_var(&self, key: &str) -> Result<Option<String>> {
            self.inner.env_var(key).await
        }

        async fn exec(
            &self,
            args: &[String],
            options: ExecOptions,
        ) -> Result<Box<dyn KaosProcess>> {
            self.inner.exec(args, options).await
        }

        async fn connect_tcp(&self, host: &str, port: u16) -> Result<Box<dyn AsyncReadWrite>> {
            self.inner.connect_tcp(host, port).await
        }
    }

    #[tokio::test]
    async fn save_and_load_mcp_config_use_kaos_share_dir() {
        let _lock = ENV_LOCK.lock().await;
        let temp_dir = TempDir::new().expect("temp dir");
        let _guard = EnvGuard::set("KIMI_SHARE_DIR", &temp_dir.path().to_string_lossy());

        let value = json!({
            "mcpServers": {
                "demo": {
                    "command": "npx",
                    "args": ["demo-mcp"]
                }
            }
        });
        save_mcp_config(&value).await.expect("save config");

        let path = get_global_mcp_config_file().await;
        assert!(path.exists(true).await);

        let loaded = load_mcp_config_file(&path).await.expect("load config");
        assert_eq!(loaded, value);
    }

    #[tokio::test]
    async fn kaos_credential_store_roundtrips_tokens() {
        let _lock = ENV_LOCK.lock().await;
        let temp_dir = TempDir::new().expect("temp dir");
        let _guard = EnvGuard::set("KIMI_SHARE_DIR", &temp_dir.path().to_string_lossy());
        let server_url = "https://example.com/mcp";
        let store = get_mcp_credential_store(server_url).await;

        assert!(!has_oauth_tokens(server_url).await.expect("no tokens yet"));

        let credentials = StoredCredentials {
            client_id: "client-id".to_string(),
            token_response: Some(
                serde_json::from_value::<OAuthTokenResponse>(
                    json!({"access_token": "token", "token_type": "Bearer"}),
                )
                .expect("token response"),
            ),
            granted_scopes: Vec::new(),
            token_received_at: None,
        };
        store.save(credentials).await.expect("save tokens");

        assert!(has_oauth_tokens(server_url).await.expect("tokens present"));

        store.clear().await.expect("clear tokens");
        assert!(!has_oauth_tokens(server_url).await.expect("tokens cleared"));
    }

    #[tokio::test]
    async fn kaos_credential_store_propagates_non_not_found_read_errors() {
        let _lock = ENV_LOCK.lock().await;
        with_current_kaos_scope(async {
            let temp_dir = TempDir::new().expect("temp dir");
            let auth_path = KaosPath::from(temp_dir.path().join("mcp_auth.json"));
            let store = KaosCredentialStore::new(auth_path.clone(), "https://example.com/mcp");
            let token = set_current_kaos(Arc::new(CredentialReadFailKaos::new(auth_path)));

            let load_err = store.load().await.expect_err("load should fail");
            assert!(load_err.to_string().contains("simulated permission denied"));

            let credentials = StoredCredentials {
                client_id: "client-id".to_string(),
                token_response: Some(
                    serde_json::from_value::<OAuthTokenResponse>(
                        json!({"access_token": "token", "token_type": "Bearer"}),
                    )
                    .expect("token response"),
                ),
                granted_scopes: Vec::new(),
                token_received_at: None,
            };
            let save_err = store.save(credentials).await.expect_err("save should fail");
            assert!(save_err.to_string().contains("simulated permission denied"));

            let clear_err = store.clear().await.expect_err("clear should fail");
            assert!(
                clear_err
                    .to_string()
                    .contains("simulated permission denied")
            );

            reset_current_kaos(token);
        })
        .await;
    }
}
