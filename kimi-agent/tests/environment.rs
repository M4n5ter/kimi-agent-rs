use std::collections::HashMap;
use std::sync::Arc;

use kaos::{
    CurrentKaosToken, Kaos, KaosPath, KaosPlatform, KaosProcess, LineStream, LocalKaos,
    StrOrKaosPath, reset_current_kaos, set_current_kaos, with_current_kaos_scope,
};
use kimi_agent::utils::Environment;
use tempfile::TempDir;

struct EnvOnlyKaos {
    inner: LocalKaos,
    env: HashMap<String, String>,
    name: &'static str,
    platform: KaosPlatform,
}

impl EnvOnlyKaos {
    fn new(name: &'static str, env: HashMap<String, String>, platform: KaosPlatform) -> Self {
        Self {
            inner: LocalKaos::new(),
            env,
            name,
            platform,
        }
    }
}

#[async_trait::async_trait]
impl Kaos for EnvOnlyKaos {
    fn name(&self) -> &str {
        self.name
    }

    fn platform(&self) -> KaosPlatform {
        self.platform.clone()
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

    async fn chdir(&self, path: &KaosPath) -> anyhow::Result<()> {
        self.inner.chdir(path).await
    }

    async fn stat(
        &self,
        path: &KaosPath,
        follow_symlinks: bool,
    ) -> anyhow::Result<kaos::StatResult> {
        self.inner.stat(path, follow_symlinks).await
    }

    async fn iterdir(&self, path: &KaosPath) -> anyhow::Result<Vec<KaosPath>> {
        self.inner.iterdir(path).await
    }

    async fn glob(
        &self,
        path: &KaosPath,
        pattern: &str,
        case_sensitive: bool,
    ) -> anyhow::Result<Vec<KaosPath>> {
        self.inner.glob(path, pattern, case_sensitive).await
    }

    async fn read_bytes(&self, path: &KaosPath, limit: Option<usize>) -> anyhow::Result<Vec<u8>> {
        self.inner.read_bytes(path, limit).await
    }

    async fn read_text(&self, path: &KaosPath) -> anyhow::Result<String> {
        self.inner.read_text(path).await
    }

    async fn read_lines(&self, path: &KaosPath) -> anyhow::Result<Vec<String>> {
        self.inner.read_lines(path).await
    }

    async fn read_lines_stream(&self, path: &KaosPath) -> anyhow::Result<LineStream> {
        self.inner.read_lines_stream(path).await
    }

    async fn write_bytes(&self, path: &KaosPath, data: &[u8]) -> anyhow::Result<usize> {
        self.inner.write_bytes(path, data).await
    }

    async fn write_text(&self, path: &KaosPath, data: &str, append: bool) -> anyhow::Result<usize> {
        self.inner.write_text(path, data, append).await
    }

    async fn chmod(&self, path: &KaosPath, mode: u32) -> anyhow::Result<()> {
        self.inner.chmod(path, mode).await
    }

    async fn mkdir(&self, path: &KaosPath, parents: bool, exist_ok: bool) -> anyhow::Result<()> {
        self.inner.mkdir(path, parents, exist_ok).await
    }

    async fn env_var(&self, key: &str) -> anyhow::Result<Option<String>> {
        Ok(self.env.get(key).cloned())
    }

    async fn exec(&self, args: &[String]) -> anyhow::Result<Box<dyn KaosProcess>> {
        self.inner.exec(args).await
    }
}

struct EnvOnlyKaosGuard {
    token: Option<CurrentKaosToken>,
}

impl EnvOnlyKaosGuard {
    fn new(kaos: EnvOnlyKaos) -> Self {
        let token = set_current_kaos(Arc::new(kaos));
        Self { token: Some(token) }
    }
}

impl Drop for EnvOnlyKaosGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_kaos(token);
        }
    }
}

#[tokio::test]
async fn test_environment_detection() {
    let env = Environment::detect().await;

    assert!(!env.os_kind.is_empty());
    assert!(!env.os_arch.is_empty());
    assert!(!env.os_version.is_empty());

    if env.os_kind == "Windows" {
        assert_eq!(env.shell_name, "Windows PowerShell");
        assert_eq!(env.shell_path.to_string_lossy(), "powershell.exe");
    } else {
        assert!(env.shell_name == "bash" || env.shell_name == "zsh" || env.shell_name == "sh");
        let shell_path = env.shell_path.to_string_lossy();
        assert!(!shell_path.is_empty());
        if env.shell_name == "bash" {
            assert!(shell_path.ends_with("bash"));
        } else if env.shell_name == "zsh" {
            assert!(shell_path.ends_with("zsh"));
        } else {
            assert!(shell_path.ends_with("sh"));
        }
    }
}

#[tokio::test]
async fn test_environment_detection_prefers_backend_shell_env_for_non_local_backend() {
    with_current_kaos_scope(async {
        let _guard = EnvOnlyKaosGuard::new(EnvOnlyKaos::new(
            "ssh",
            HashMap::from([("SHELL".to_string(), "/bin/bash".to_string())]),
            KaosPlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                abi: None,
                libc: Some("gnu".to_string()),
            },
        ));

        let env = Environment::detect().await;
        assert_eq!(env.os_kind, "Linux");
        assert_eq!(env.os_version, "");
        assert_eq!(env.shell_name, "bash");
        assert_eq!(env.shell_path.to_string_lossy(), "/bin/bash");
    })
    .await;
}

#[tokio::test]
async fn test_environment_detection_accepts_backend_zsh_env() {
    let temp = TempDir::new().expect("temp dir");
    let zsh_path = temp.path().join("zsh");
    std::fs::write(&zsh_path, "#!/bin/zsh\n").expect("write fake zsh");

    with_current_kaos_scope(async {
        let _guard = EnvOnlyKaosGuard::new(EnvOnlyKaos::new(
            "ssh",
            HashMap::from([("SHELL".to_string(), zsh_path.to_string_lossy().to_string())]),
            KaosPlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                abi: None,
                libc: Some("gnu".to_string()),
            },
        ));

        let env = Environment::detect().await;
        assert_eq!(env.shell_name, "zsh");
        assert_eq!(env.shell_path.to_string_lossy(), zsh_path.to_string_lossy());
    })
    .await;
}

#[tokio::test]
async fn test_environment_detection_ignores_backend_sh_env_and_keeps_preferred_fallbacks() {
    with_current_kaos_scope(async {
        let _guard = EnvOnlyKaosGuard::new(EnvOnlyKaos::new(
            "ssh",
            HashMap::from([("SHELL".to_string(), "/bin/sh".to_string())]),
            KaosPlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                abi: None,
                libc: Some("gnu".to_string()),
            },
        ));

        let env = Environment::detect().await;
        let expected = if std::path::Path::new("/bin/bash").is_file()
            || std::path::Path::new("/usr/bin/bash").is_file()
            || std::path::Path::new("/usr/local/bin/bash").is_file()
        {
            "bash"
        } else if std::path::Path::new("/bin/zsh").is_file()
            || std::path::Path::new("/usr/bin/zsh").is_file()
            || std::path::Path::new("/usr/local/bin/zsh").is_file()
        {
            "zsh"
        } else {
            "sh"
        };
        assert_eq!(env.shell_name, expected);
    })
    .await;
}

#[tokio::test]
async fn test_environment_detection_ignores_unsupported_backend_shell_env() {
    with_current_kaos_scope(async {
        let _guard = EnvOnlyKaosGuard::new(EnvOnlyKaos::new(
            "ssh",
            HashMap::from([("SHELL".to_string(), "/usr/bin/fish".to_string())]),
            KaosPlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                abi: None,
                libc: Some("gnu".to_string()),
            },
        ));

        let env = Environment::detect().await;
        assert!(env.shell_name == "bash" || env.shell_name == "zsh" || env.shell_name == "sh");
    })
    .await;
}
