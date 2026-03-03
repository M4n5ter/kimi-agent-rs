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

    match legacy_mcp_auth_contains(server_url).await {
        Ok(Some(path)) => {
            eprintln!("{}", legacy_mcp_auth_warning(&path, &current_arg0()));
        }
        Ok(None) => {}
        Err(err) => {
            eprintln!(
                "Warning: failed to probe ignored legacy MCP auth file: {err}. SQLite storage is authoritative now. Re-run `{} mcp auth <server>` if authorization is required.",
                current_arg0(),
            );
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use kaos::{
        AsyncReadWrite, ExecOptions, Kaos, KaosPath, KaosPlatform, KaosProcess, LineStream,
        LocalKaos, StrOrKaosPath, reset_current_kaos, set_current_kaos, with_current_kaos_scope,
    };
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

    struct FixedHomeKaos {
        inner: LocalKaos,
        home: KaosPath,
    }

    impl FixedHomeKaos {
        fn new(home: KaosPath) -> Self {
            Self {
                inner: LocalKaos::new(),
                home,
            }
        }
    }

    #[async_trait::async_trait]
    impl Kaos for FixedHomeKaos {
        fn name(&self) -> &str {
            "fixed-home"
        }

        fn storage_name(&self) -> String {
            "root@example.com:22".to_string()
        }

        fn platform(&self) -> KaosPlatform {
            self.inner.platform()
        }

        fn normpath(&self, path: &StrOrKaosPath<'_>) -> KaosPath {
            self.inner.normpath(path)
        }

        fn home(&self) -> KaosPath {
            self.home.clone()
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

        async fn env_var(&self, _key: &str) -> Result<Option<String>> {
            Ok(None)
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

    #[tokio::test]
    async fn malformed_legacy_auth_file_does_not_break_oauth_token_checks() {
        let temp_dir = TempDir::new().expect("temp dir");
        let storage = open_test_storage(&temp_dir).await;
        let auth_path = temp_dir
            .path()
            .join(".kimi")
            .join("credentials")
            .join("mcp_auth.json");
        std::fs::create_dir_all(auth_path.parent().expect("auth parent")).expect("create auth dir");
        std::fs::write(&auth_path, "{invalid json").expect("write malformed auth file");

        let kaos = Arc::new(FixedHomeKaos::new(KaosPath::from(
            temp_dir.path().to_path_buf(),
        )));
        with_current_kaos_scope(async move {
            let token = set_current_kaos(kaos);
            let has_tokens =
                has_oauth_tokens(&storage, &KaosConfig::Local, "https://example.com/mcp")
                    .await
                    .expect("legacy auth probe should not fail");
            reset_current_kaos(token);

            assert!(!has_tokens);
        })
        .await;
    }
}
