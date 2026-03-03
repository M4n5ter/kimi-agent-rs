use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use kaos::{KaosPath, get_current_kaos, is_not_found_error};
use rmcp::transport::auth::StoredCredentials;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct LegacyCredentialFile {
    #[serde(default)]
    servers: HashMap<String, StoredCredentials>,
}

pub fn current_arg0() -> String {
    std::env::args_os()
        .next()
        .and_then(|arg0| {
            let path = PathBuf::from(arg0);
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .filter(|arg0| !arg0.is_empty())
        .unwrap_or_else(|| "kimi-agent".to_string())
}

async fn legacy_mcp_state_dir() -> Result<KaosPath> {
    match kaos::env_var("KIMI_SHARE_DIR").await {
        Ok(Some(path)) if !path.trim().is_empty() => Ok(KaosPath::new(&path)),
        Ok(_) => Ok(get_current_kaos().app_state_dir("kimi")),
        Err(err) => Err(err).context("resolve legacy MCP state dir"),
    }
}

pub async fn legacy_mcp_config_path() -> Result<KaosPath> {
    Ok(legacy_mcp_state_dir().await? / "mcp.json")
}

pub async fn legacy_mcp_auth_path() -> Result<KaosPath> {
    Ok(legacy_mcp_state_dir().await? / "credentials" / "mcp_auth.json")
}

pub async fn legacy_mcp_config_exists() -> Result<Option<KaosPath>> {
    let path = legacy_mcp_config_path().await?;
    if path.exists(true).await {
        Ok(Some(path))
    } else {
        Ok(None)
    }
}

pub async fn legacy_mcp_auth_contains(server_key: &str) -> Result<Option<KaosPath>> {
    let path = legacy_mcp_auth_path().await?;
    let text = match path.read_text().await {
        Ok(text) => text,
        Err(err) if is_not_found_error(&err) => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("read {}", path.to_string_lossy())),
    };

    let credentials = serde_json::from_str::<LegacyCredentialFile>(&text)
        .with_context(|| format!("parse {}", path.to_string_lossy()))?;
    Ok(credentials.servers.contains_key(server_key).then_some(path))
}

pub fn legacy_mcp_config_warning(path: &KaosPath, arg0: &str) -> String {
    format!(
        "Warning: legacy MCP config {} is ignored. SQLite storage is authoritative now. Use `{} mcp ...` to manage MCP servers and auth.",
        path.to_string_lossy(),
        arg0,
    )
}

pub fn legacy_mcp_auth_warning(path: &KaosPath, arg0: &str) -> String {
    format!(
        "Warning: legacy MCP auth file {} is ignored. SQLite storage is authoritative now. Re-run `{} mcp auth <server>` to authorize MCP servers again.",
        path.to_string_lossy(),
        arg0,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use kaos::{
        AsyncReadWrite, ExecOptions, Kaos, KaosPath, KaosPlatform, KaosProcess, LineStream,
        LocalKaos, StrOrKaosPath, reset_current_kaos, set_current_kaos, with_current_kaos_scope,
    };
    use tempfile::TempDir;

    use super::{
        legacy_mcp_auth_contains, legacy_mcp_auth_path, legacy_mcp_config_exists,
        legacy_mcp_config_path,
    };

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

    #[tokio::test]
    async fn legacy_mcp_paths_follow_current_kaos_scope() {
        let home = TempDir::new().expect("home dir");
        let kaos = Arc::new(FixedHomeKaos::new(KaosPath::from(
            home.path().to_path_buf(),
        )));

        with_current_kaos_scope(async move {
            let token = set_current_kaos(kaos);
            let config_path = legacy_mcp_config_path().await.expect("config path");
            let auth_path = legacy_mcp_auth_path().await.expect("auth path");
            reset_current_kaos(token);

            assert_eq!(
                config_path,
                KaosPath::from(home.path().join(".kimi").join("mcp.json"))
            );
            assert_eq!(
                auth_path,
                KaosPath::from(
                    home.path()
                        .join(".kimi")
                        .join("credentials")
                        .join("mcp_auth.json")
                )
            );
        })
        .await;
    }

    #[tokio::test]
    async fn legacy_mcp_auth_probe_detects_remote_server_tokens() {
        let home = TempDir::new().expect("home dir");
        let auth_path = home
            .path()
            .join(".kimi")
            .join("credentials")
            .join("mcp_auth.json");
        std::fs::create_dir_all(auth_path.parent().expect("auth parent")).expect("create auth dir");
        std::fs::write(
            &auth_path,
            serde_json::json!({
                "servers": {
                    "https://example.com/mcp": {
                        "client_id": "client-id",
                        "token_response": {"access_token": "token", "token_type": "Bearer"},
                        "granted_scopes": []
                    }
                }
            })
            .to_string(),
        )
        .expect("write auth file");

        let kaos = Arc::new(FixedHomeKaos::new(KaosPath::from(
            home.path().to_path_buf(),
        )));
        with_current_kaos_scope(async move {
            let token = set_current_kaos(kaos);
            let found = legacy_mcp_auth_contains("https://example.com/mcp")
                .await
                .expect("probe auth");
            let config = legacy_mcp_config_exists().await.expect("probe config");
            reset_current_kaos(token);

            assert!(config.is_none());
            assert_eq!(found, Some(KaosPath::from(auth_path)));
        })
        .await;
    }
}
