use std::collections::HashMap;

use kaos::{KaosPath, get_current_kaos};
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
}

#[async_trait::async_trait]
impl CredentialStore for KaosCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let text = match self.path.read_text().await {
            Ok(text) => text,
            Err(err) if !self.path.exists(true).await => {
                return Ok(None);
            }
            Err(err) => {
                return Err(AuthError::InternalError(format!(
                    "Failed to read MCP auth file: {err}"
                )));
            }
        };
        let data: McpCredentialFile = serde_json::from_str(&text)
            .map_err(|err| AuthError::InternalError(format!("Invalid MCP auth file: {err}")))?;
        Ok(data.servers.get(&self.server_key).cloned())
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let mut data = match self.path.read_text().await {
            Ok(text) => serde_json::from_str::<McpCredentialFile>(&text)
                .map_err(|err| AuthError::InternalError(format!("Invalid MCP auth file: {err}")))?,
            Err(err) if !self.path.exists(true).await => McpCredentialFile::default(),
            Err(err) => {
                return Err(AuthError::InternalError(format!(
                    "Failed to read MCP auth file: {err}"
                )));
            }
        };
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
        let mut data = match self.path.read_text().await {
            Ok(text) => serde_json::from_str::<McpCredentialFile>(&text)
                .map_err(|err| AuthError::InternalError(format!("Invalid MCP auth file: {err}")))?,
            Err(err) if !self.path.exists(true).await => McpCredentialFile::default(),
            Err(err) => {
                return Err(AuthError::InternalError(format!(
                    "Failed to read MCP auth file: {err}"
                )));
            }
        };
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
    use std::sync::LazyLock;

    use rmcp::transport::auth::{CredentialStore, OAuthTokenResponse, StoredCredentials};
    use serde_json::json;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use super::{
        get_global_mcp_config_file, get_mcp_credential_store, has_oauth_tokens,
        load_mcp_config_file, save_mcp_config,
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
}
