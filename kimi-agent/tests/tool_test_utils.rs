use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use kaos::{
    AsyncReadWrite, CurrentKaosToken, ExecOptions, Kaos, KaosPath, LocalKaos, reset_current_kaos,
    set_current_kaos,
};
use kimi_agent::config::{
    ModelCapability, MoonshotFetchConfig, MoonshotSearchConfig, StorageConfig, get_default_config,
};
use kimi_agent::llm::LLM;
use kimi_agent::session::Session;
use kimi_agent::soul::agent::{BuiltinSystemPromptArgs, LaborMarket, Runtime};
use kimi_agent::soul::approval::Approval;
use kimi_agent::soul::denwarenji::DenwaRenji;
use kimi_agent::storage::Storage;
use kimi_agent::utils::Environment;
use kosong::chat_provider::echo::EchoChatProvider;
use tempfile::TempDir;

pub struct RuntimeFixture {
    pub runtime: Runtime,
    _work_dir: TempDir,
    _storage_dir: TempDir,
}

fn block_on_test_future<F>(future: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(future))
            }
            tokio::runtime::RuntimeFlavor::CurrentThread => std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .expect("build test runtime")
                            .block_on(future)
                    })
                    .join()
                    .expect("join scoped test runtime")
            }),
            _ => std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .expect("build test runtime")
                            .block_on(future)
                    })
                    .join()
                    .expect("join scoped test runtime")
            }),
        }
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build test runtime")
            .block_on(future)
    }
}

impl RuntimeFixture {
    #[allow(dead_code)]
    pub fn new() -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(ModelCapability::ImageIn);
        capabilities.insert(ModelCapability::VideoIn);
        Self::with_capabilities(capabilities)
    }

    pub fn with_capabilities(capabilities: HashSet<ModelCapability>) -> Self {
        let work_dir = TempDir::new().expect("temp work dir");
        let storage_dir = TempDir::new().expect("temp storage dir");

        let work_path = KaosPath::from(PathBuf::from(work_dir.path()));
        let mut config = get_default_config();
        config.storage = StorageConfig {
            database_path: storage_dir.path().join("state.db").display().to_string(),
            busy_timeout_ms: 1_000,
        };
        config.services.moonshot_search = Some(MoonshotSearchConfig {
            base_url: "https://api.kimi.com/coding/v1/search".to_string(),
            api_key: "test-api-key".to_string(),
            custom_headers: None,
        });
        config.services.moonshot_fetch = Some(MoonshotFetchConfig {
            base_url: "https://api.kimi.com/coding/v1/fetch".to_string(),
            api_key: "test-api-key".to_string(),
            custom_headers: None,
        });
        let storage =
            block_on_test_future(Storage::open(&config.storage)).expect("open test storage");
        let mut session = block_on_test_future(Session::create(
            storage.clone(),
            config.kaos.clone(),
            work_path.clone(),
            Some("test".to_string()),
        ))
        .expect("create test session");
        session.title = "Test Session".to_string();

        let llm = LLM {
            chat_provider: Box::new(EchoChatProvider),
            max_context_size: 100_000,
            capabilities,
            model_config: None,
            provider_config: None,
        };

        let environment = Environment {
            os_kind: if cfg!(windows) { "Windows" } else { "Unix" }.to_string(),
            os_arch: "x86_64".to_string(),
            os_version: "1.0".to_string(),
            shell_name: if cfg!(windows) {
                "Windows PowerShell"
            } else {
                "bash"
            }
            .to_string(),
            shell_path: if cfg!(windows) {
                KaosPath::from(PathBuf::from("powershell.exe"))
            } else {
                KaosPath::from(PathBuf::from("/bin/bash"))
            },
        };

        let runtime = Runtime {
            config,
            storage: storage.clone(),
            llm: Some(Arc::new(llm)),
            session: session.clone(),
            builtin_args: BuiltinSystemPromptArgs {
                KIMI_NOW: "1970-01-01T00:00:00+00:00".to_string(),
                KIMI_WORK_DIR: work_path,
                KIMI_WORK_DIR_LS: "Test ls content".to_string(),
                KIMI_AGENTS_MD: "Test agents content".to_string(),
                KIMI_SKILLS: "No skills found.".to_string(),
            },
            denwa_renji: Arc::new(tokio::sync::Mutex::new(DenwaRenji::new())),
            approval: Arc::new(Approval::new(true)),
            labor_market: Arc::new(tokio::sync::Mutex::new(LaborMarket::new())),
            environment,
            skills: Default::default(),
        };

        Self {
            runtime,
            _work_dir: work_dir,
            _storage_dir: storage_dir,
        }
    }
}

impl Default for RuntimeFixture {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
pub struct TestKaos {
    inner: LocalKaos,
    cwd: Mutex<KaosPath>,
}

#[allow(dead_code)]
impl TestKaos {
    pub fn new(cwd: KaosPath) -> Self {
        Self {
            inner: LocalKaos::new(),
            cwd: Mutex::new(cwd),
        }
    }
}

#[async_trait::async_trait]
impl Kaos for TestKaos {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn platform(&self) -> kaos::KaosPlatform {
        self.inner.platform()
    }

    fn normpath(&self, path: &kaos::StrOrKaosPath<'_>) -> KaosPath {
        self.inner.normpath(path)
    }

    fn home(&self) -> KaosPath {
        self.inner.home()
    }

    fn cwd(&self) -> KaosPath {
        self.cwd.lock().unwrap().clone()
    }

    async fn chdir(&self, path: &KaosPath) -> anyhow::Result<()> {
        *self.cwd.lock().unwrap() = path.clone();
        Ok(())
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

    async fn read_lines_stream(&self, path: &KaosPath) -> anyhow::Result<kaos::LineStream> {
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
        self.inner.env_var(key).await
    }

    async fn exec(
        &self,
        args: &[String],
        options: ExecOptions,
    ) -> anyhow::Result<Box<dyn kaos::KaosProcess>> {
        self.inner.exec(args, options).await
    }

    async fn connect_tcp(&self, host: &str, port: u16) -> anyhow::Result<Box<dyn AsyncReadWrite>> {
        self.inner.connect_tcp(host, port).await
    }
}

#[allow(dead_code)]
pub struct TestKaosGuard {
    token: Option<CurrentKaosToken>,
}

#[allow(dead_code)]
impl TestKaosGuard {
    pub fn new(cwd: KaosPath) -> Self {
        let test_kaos = Arc::new(TestKaos::new(cwd));
        let token = set_current_kaos(test_kaos);
        Self { token: Some(token) }
    }
}

impl Drop for TestKaosGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_kaos(token);
        }
    }
}

pub fn normalize_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[test]
fn test_normalize_newlines() {
    assert_eq!(normalize_newlines("a\r\nb\rc\n"), "a\nb\nc\n");
}
