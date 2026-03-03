use rmcp::transport::auth::{AuthError, CredentialStore, StoredCredentials};
use serde_json::{Map, Value, json};

use crate::config::KaosConfig;
use crate::exception::MCPConfigError;
use crate::mcp_legacy::{current_arg0, legacy_mcp_auth_contains, legacy_mcp_auth_warning};
use crate::storage::Storage;

pub async fn load_mcp_config_file(path: &kaos::KaosPath) -> Result<Value, MCPConfigError> {
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
    map.get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| MCPConfigError::new("mcpServers must be a JSON object"))
}

pub async fn load_persisted_mcp_config(
    storage: &Storage,
    kaos: &KaosConfig,
) -> Result<Value, MCPConfigError> {
    let records = storage
        .load_mcp_servers(kaos)
        .await
        .map_err(|err| MCPConfigError::new(format!("Failed to load MCP config: {err}")))?;
    let servers = records
        .into_iter()
        .map(|record| (record.name, record.config))
        .collect::<Map<String, Value>>();
    Ok(json!({ "mcpServers": servers }))
}

pub async fn save_persisted_mcp_config(
    storage: &Storage,
    kaos: &KaosConfig,
    value: &Value,
) -> Result<(), MCPConfigError> {
    let mut normalized = value.clone();
    let servers = ensure_mcp_servers(&mut normalized)?;
    let desired = servers
        .iter()
        .map(|(name, config)| (name.clone(), config.clone()))
        .collect::<Vec<_>>();
    storage
        .replace_mcp_servers(kaos, &desired)
        .await
        .map_err(|err| MCPConfigError::new(format!("Failed to save MCP config: {err}")))
}

#[derive(Debug, Clone)]
pub struct KaosCredentialStore {
    storage: Storage,
    kaos: KaosConfig,
    server_key: String,
}

impl KaosCredentialStore {
    pub fn new(storage: Storage, kaos: KaosConfig, server_key: impl Into<String>) -> Self {
        Self {
            storage,
            kaos,
            server_key: server_key.into(),
        }
    }
}

#[async_trait::async_trait]
impl CredentialStore for KaosCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        self.storage
            .load_mcp_credentials(&self.kaos, &self.server_key)
            .await
            .map_err(|err| {
                AuthError::InternalError(format!("Failed to read MCP credentials: {err}"))
            })
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        self.storage
            .save_mcp_credentials(&self.kaos, &self.server_key, &credentials)
            .await
            .map_err(|err| {
                AuthError::InternalError(format!("Failed to save MCP credentials: {err}"))
            })
    }

    async fn clear(&self) -> Result<(), AuthError> {
        self.storage
            .clear_mcp_credentials(&self.kaos, &self.server_key)
            .await
            .map_err(|err| {
                AuthError::InternalError(format!("Failed to clear MCP credentials: {err}"))
            })
    }
}

pub fn get_mcp_credential_store(
    storage: &Storage,
    kaos: &KaosConfig,
    server_url: &str,
) -> KaosCredentialStore {
    KaosCredentialStore::new(storage.clone(), kaos.clone(), server_url)
}

pub async fn has_oauth_tokens(
    storage: &Storage,
    kaos: &KaosConfig,
    server_url: &str,
) -> Result<bool, AuthError> {
    let has_tokens = get_mcp_credential_store(storage, kaos, server_url)
        .load()
        .await?
        .and_then(|creds| creds.token_response)
        .is_some();
    if has_tokens {
        return Ok(true);
    }

    if let Some(path) = legacy_mcp_auth_contains(server_url).await.map_err(|err| {
        AuthError::InternalError(format!("Failed to probe legacy MCP auth file: {err}"))
    })? {
        eprintln!("{}", legacy_mcp_auth_warning(&path, &current_arg0()));
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rmcp::transport::auth::{CredentialStore, OAuthTokenResponse, StoredCredentials};
    use serde_json::json;
    use tempfile::TempDir;

    use super::{
        get_mcp_credential_store, has_oauth_tokens, load_persisted_mcp_config,
        save_persisted_mcp_config,
    };
    use crate::config::{KaosConfig, StorageConfig};
    use crate::storage::Storage;

    fn test_ssh_kaos() -> KaosConfig {
        KaosConfig::Ssh {
            options: kaos::SshKaosOptions {
                host: "example.com".to_string(),
                port: 22,
                username: Some("root".to_string()),
                password: None,
                key_paths: Vec::new(),
                key_contents: Vec::new(),
                cwd: Some("/root".to_string()),
                known_hosts_path: None,
                host_key_policy: kaos::SshHostKeyPolicy::AcceptNew,
                connect_timeout_seconds: 15,
            },
        }
    }

    async fn open_test_storage(temp_dir: &TempDir) -> Storage {
        Storage::open(&StorageConfig {
            database_path: temp_dir.path().join("state.db").display().to_string(),
            busy_timeout_ms: 1_000,
        })
        .await
        .expect("open storage")
    }

    #[tokio::test]
    async fn save_and_load_mcp_config_use_storage() {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = open_test_storage(&temp_dir).await;
        let value = json!({
            "mcpServers": {
                "demo": {
                    "command": "npx",
                    "args": ["demo-mcp"]
                }
            }
        });

        save_persisted_mcp_config(&storage, &KaosConfig::Local, &value)
            .await
            .expect("save config");
        let loaded = load_persisted_mcp_config(&storage, &KaosConfig::Local)
            .await
            .expect("load config");
        assert_eq!(loaded, value);
    }

    #[tokio::test]
    async fn kaos_credential_store_roundtrips_tokens() {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = open_test_storage(&temp_dir).await;
        let server_url = "https://example.com/mcp";
        let store = get_mcp_credential_store(&storage, &KaosConfig::Local, server_url);

        assert!(
            !has_oauth_tokens(&storage, &KaosConfig::Local, server_url)
                .await
                .expect("no tokens yet")
        );

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

        assert!(
            has_oauth_tokens(&storage, &KaosConfig::Local, server_url)
                .await
                .expect("tokens present")
        );

        store.clear().await.expect("clear tokens");
        assert!(
            !has_oauth_tokens(&storage, &KaosConfig::Local, server_url)
                .await
                .expect("tokens cleared")
        );
    }

    #[tokio::test]
    async fn mcp_config_isolated_by_kaos_scope() -> Result<()> {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = open_test_storage(&temp_dir).await;

        save_persisted_mcp_config(
            &storage,
            &KaosConfig::Local,
            &json!({"mcpServers": {"local-demo": {"command": "demo"}}}),
        )
        .await?;
        save_persisted_mcp_config(
            &storage,
            &test_ssh_kaos(),
            &json!({"mcpServers": {"ssh-demo": {"command": "demo"}}}),
        )
        .await?;

        let local = load_persisted_mcp_config(&storage, &KaosConfig::Local).await?;
        let remote = load_persisted_mcp_config(&storage, &test_ssh_kaos()).await?;

        assert!(local["mcpServers"].get("local-demo").is_some());
        assert!(local["mcpServers"].get("ssh-demo").is_none());
        assert!(remote["mcpServers"].get("ssh-demo").is_some());
        assert!(remote["mcpServers"].get("local-demo").is_none());
        Ok(())
    }
}
