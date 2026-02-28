use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::anyhow;
use kaos::{
    CurrentKaosToken, Kaos, KaosPath, KaosProcess, LineStream, LocalKaos, StrOrKaosPath,
    reset_current_kaos, set_current_kaos, with_current_kaos_scope,
};
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::Mutex;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use kimi_agent::config::{LLMModel, LLMProvider, ModelCapability, ProviderType};
use kimi_agent::llm::{augment_provider_with_env_vars, create_llm};
use kosong::chat_provider::anthropic::Anthropic;
use kosong::chat_provider::echo::EchoChatProvider;
use kosong::chat_provider::kimi::Kimi;
use kosong::chat_provider::openai_legacy::OpenAILegacy;
use kosong::chat_provider::openai_responses::OpenAIResponses;

static ENV_LOCK: Mutex<()> = Mutex::const_new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::set_var(self.key, prev);
            }
        } else {
            // SAFETY: tests serialize env access via ENV_LOCK to avoid races.
            unsafe {
                std::env::remove_var(self.key);
            }
        }
    }
}

struct BackendEnvKaos {
    inner: LocalKaos,
    env: HashMap<String, String>,
    failing_keys: HashSet<String>,
}

impl BackendEnvKaos {
    fn with_failing_keys(env: HashMap<String, String>, failing_keys: HashSet<String>) -> Self {
        Self {
            inner: LocalKaos::new(),
            env,
            failing_keys,
        }
    }
}

#[async_trait::async_trait]
impl Kaos for BackendEnvKaos {
    fn name(&self) -> &str {
        "test-backend"
    }

    fn platform(&self) -> kaos::KaosPlatform {
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
        if self.failing_keys.contains(key) {
            return Err(anyhow!("backend env lookup failed for `{key}`"));
        }
        Ok(self.env.get(key).cloned())
    }

    async fn exec(&self, args: &[String]) -> anyhow::Result<Box<dyn KaosProcess>> {
        self.inner.exec(args).await
    }
}

struct BackendEnvKaosGuard {
    token: Option<CurrentKaosToken>,
}

impl BackendEnvKaosGuard {
    fn new(env: HashMap<String, String>) -> Self {
        Self::with_failing_keys(env, HashSet::new())
    }

    fn with_failing_keys(env: HashMap<String, String>, failing_keys: HashSet<String>) -> Self {
        let kaos = Arc::new(BackendEnvKaos::with_failing_keys(env, failing_keys));
        let token = set_current_kaos(kaos);
        Self { token: Some(token) }
    }
}

impl Drop for BackendEnvKaosGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_kaos(token);
        }
    }
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_kimi() {
    let _lock = ENV_LOCK.lock().await;
    let _guards = [
        EnvGuard::set("KIMI_BASE_URL", "https://env.test/v1"),
        EnvGuard::set("KIMI_API_KEY", "env-key"),
        EnvGuard::set("KIMI_MODEL_NAME", "kimi-env-model"),
        EnvGuard::set("KIMI_MODEL_MAX_CONTEXT_SIZE", "8192"),
        EnvGuard::set("KIMI_MODEL_CAPABILITIES", "Image_In,THINKING,unknown"),
    ];

    let mut provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: String::new(),
        api_key: String::new(),
        env: None,
        custom_headers: None,
    };
    let mut model = LLMModel {
        provider: "kimi".to_string(),
        model: String::new(),
        max_context_size: 0,
        capabilities: None,
    };

    augment_provider_with_env_vars(&mut provider, &mut model)
        .await
        .expect("env overrides");

    assert_eq!(
        provider,
        LLMProvider {
            provider_type: ProviderType::Kimi,
            base_url: "https://env.test/v1".to_string(),
            api_key: "env-key".to_string(),
            env: None,
            custom_headers: None,
        }
    );
    assert_eq!(
        model,
        LLMModel {
            provider: "kimi".to_string(),
            model: "kimi-env-model".to_string(),
            max_context_size: 8192,
            capabilities: Some(HashSet::from([
                ModelCapability::ImageIn,
                ModelCapability::Thinking,
            ])),
        }
    );
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_preserves_explicit_kimi_config() {
    let _lock = ENV_LOCK.lock().await;
    let _guards = [
        EnvGuard::set("KIMI_BASE_URL", "https://env.test/v1"),
        EnvGuard::set("KIMI_API_KEY", "env-key"),
        EnvGuard::set("KIMI_MODEL_NAME", "kimi-env-model"),
        EnvGuard::set("KIMI_MODEL_MAX_CONTEXT_SIZE", "8192"),
        EnvGuard::set("KIMI_MODEL_CAPABILITIES", "Image_In,THINKING"),
    ];

    let mut provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: "https://configured.test/v1".to_string(),
        api_key: "configured-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let mut model = LLMModel {
        provider: "kimi".to_string(),
        model: "kimi-configured".to_string(),
        max_context_size: 4096,
        capabilities: Some(HashSet::from([ModelCapability::VideoIn])),
    };

    let applied = augment_provider_with_env_vars(&mut provider, &mut model)
        .await
        .expect("env overrides");

    assert!(applied.is_empty());
    assert_eq!(provider.base_url, "https://configured.test/v1");
    assert_eq!(provider.api_key, "configured-key");
    assert_eq!(model.model, "kimi-configured");
    assert_eq!(model.max_context_size, 4096);
    assert_eq!(
        model.capabilities,
        Some(HashSet::from([ModelCapability::VideoIn]))
    );
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_invalid_max_context_size() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("KIMI_MODEL_MAX_CONTEXT_SIZE", "not-a-number");

    let mut provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: "https://original.test/v1".to_string(),
        api_key: "orig-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let mut model = LLMModel {
        provider: "kimi".to_string(),
        model: "kimi-base".to_string(),
        max_context_size: 0,
        capabilities: None,
    };

    let err = augment_provider_with_env_vars(&mut provider, &mut model)
        .await
        .expect_err("invalid max context size");
    assert!(
        err.to_string()
            .contains("invalid literal for int() with base 10")
    );
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_anthropic_uses_backend_env() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "ANTHROPIC_BASE_URL".to_string(),
                "https://backend.anthropic.test".to_string(),
            ),
            (
                "ANTHROPIC_API_KEY".to_string(),
                "backend-anthropic-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::Anthropic,
            base_url: String::new(),
            api_key: String::new(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            max_context_size: 200_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "https://backend.anthropic.test");
        assert_eq!(provider.api_key, "backend-anthropic-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_preserves_explicit_anthropic_config() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "ANTHROPIC_BASE_URL".to_string(),
                "https://backend.anthropic.test".to_string(),
            ),
            (
                "ANTHROPIC_API_KEY".to_string(),
                "backend-anthropic-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::Anthropic,
            base_url: "https://configured.anthropic.test".to_string(),
            api_key: "configured-anthropic-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            max_context_size: 200_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "https://configured.anthropic.test");
        assert_eq!(provider.api_key, "configured-anthropic-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_skips_backend_probe_for_explicit_credentials() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::with_failing_keys(
            HashMap::new(),
            HashSet::from([
                "ANTHROPIC_BASE_URL".to_string(),
                "ANTHROPIC_API_KEY".to_string(),
            ]),
        );

        let mut provider = LLMProvider {
            provider_type: ProviderType::Anthropic,
            base_url: "https://configured.anthropic.test".to_string(),
            api_key: "configured-anthropic-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            max_context_size: 200_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("explicit config should not probe failing backend env");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "https://configured.anthropic.test");
        assert_eq!(provider.api_key, "configured-anthropic-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_google_genai_uses_gemini_env_family() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "GEMINI_BASE_URL".to_string(),
                "https://backend.gemini.test/v1beta/openai".to_string(),
            ),
            (
                "GEMINI_API_KEY".to_string(),
                "backend-gemini-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::GoogleGenai,
            base_url: String::new(),
            api_key: String::new(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "google".to_string(),
            model: "gemini-2.5-flash".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(
            provider.base_url,
            "https://backend.gemini.test/v1beta/openai"
        );
        assert_eq!(provider.api_key, "backend-gemini-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_preserves_explicit_google_genai_config() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "GEMINI_BASE_URL".to_string(),
                "https://backend.gemini.test/v1beta/openai".to_string(),
            ),
            (
                "GEMINI_API_KEY".to_string(),
                "backend-gemini-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::GoogleGenai,
            base_url: "https://configured.gemini.test/v1beta/openai".to_string(),
            api_key: "configured-gemini-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "google".to_string(),
            model: "gemini-2.5-flash".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(
            provider.base_url,
            "https://configured.gemini.test/v1beta/openai"
        );
        assert_eq!(provider.api_key, "configured-gemini-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_openai_responses_uses_openai_env_family() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "OPENAI_BASE_URL".to_string(),
                "https://backend.openai.test/v1".to_string(),
            ),
            (
                "OPENAI_API_KEY".to_string(),
                "backend-openai-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::OpenaiResponses,
            base_url: String::new(),
            api_key: String::new(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "openai-responses".to_string(),
            model: "gpt-5-codex".to_string(),
            max_context_size: 200_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "https://backend.openai.test/v1");
        assert_eq!(provider.api_key, "backend-openai-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_preserves_explicit_vertexai_config() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([
            (
                "VERTEXAI_BASE_URL".to_string(),
                "https://backend.vertex.test/v1".to_string(),
            ),
            (
                "VERTEXAI_API_KEY".to_string(),
                "backend-vertex-key".to_string(),
            ),
        ]));

        let mut provider = LLMProvider {
            provider_type: ProviderType::Vertexai,
            base_url: "https://configured.vertex.test/v1".to_string(),
            api_key: "configured-vertex-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "vertex".to_string(),
            model: "gemini-3-pro-preview".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "https://configured.vertex.test/v1");
        assert_eq!(provider.api_key, "configured-vertex-key");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_uses_backend_env_without_process_fallback() {
    let _lock = ENV_LOCK.lock().await;
    let _guards = [
        EnvGuard::set("KIMI_BASE_URL", "https://host.test/v1"),
        EnvGuard::set("KIMI_API_KEY", "host-key"),
    ];

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::new());

        let mut provider = LLMProvider {
            provider_type: ProviderType::Kimi,
            base_url: String::new(),
            api_key: String::new(),
            env: None,
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "kimi".to_string(),
            model: "kimi-base".to_string(),
            max_context_size: 4096,
            capabilities: None,
        };

        let applied = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        assert!(applied.is_empty());
        assert_eq!(provider.base_url, "");
        assert_eq!(provider.api_key, "");
    })
    .await;
}

#[tokio::test]
async fn test_augment_provider_with_env_vars_prefers_provider_env_over_backend_env() {
    let _lock = ENV_LOCK.lock().await;
    let mock_server = MockServer::start().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([(
            "VERTEXAI_API_KEY".to_string(),
            "backend-key".to_string(),
        )]));

        let base_url = format!("{}/v1", mock_server.uri());
        let api_key = "provider-key".to_string();

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer provider-key"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut provider = LLMProvider {
            provider_type: ProviderType::Vertexai,
            base_url: base_url.clone(),
            api_key: String::new(),
            env: Some(HashMap::from([(
                "VERTEXAI_API_KEY".to_string(),
                api_key.clone(),
            )])),
            custom_headers: None,
        };
        let mut model = LLMModel {
            provider: "vertex".to_string(),
            model: "gemini-3-pro-preview".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .expect("env overrides");

        let llm = create_llm(&provider, &model, None, None)
            .await
            .expect("create llm")
            .expect("llm");

        let tools: Vec<kosong::tooling::Tool> = vec![];
        let history: Vec<kosong::message::Message> = vec![];
        let _stream = llm
            .chat_provider
            .generate("", &tools, &history)
            .await
            .expect("generate request should use provider env values");
    })
    .await;
}

#[tokio::test]
async fn test_create_llm_kimi_model_parameters() {
    let _lock = ENV_LOCK.lock().await;
    let _guards = [
        EnvGuard::set("KIMI_MODEL_TEMPERATURE", "0.2"),
        EnvGuard::set("KIMI_MODEL_TOP_P", "0.8"),
        EnvGuard::set("KIMI_MODEL_MAX_TOKENS", "1234"),
    ];

    let provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: "https://api.test/v1".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "kimi".to_string(),
        model: "kimi-base".to_string(),
        max_context_size: 4096,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    let kimi = llm
        .chat_provider
        .as_any()
        .downcast_ref::<Kimi>()
        .expect("kimi provider");

    assert_eq!(
        serde_json::Value::Object(kimi.model_parameters()),
        json!({
            "base_url": "https://api.test/v1/",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 1234
        })
    );
}

#[tokio::test]
async fn test_create_llm_invalid_temperature_env() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("KIMI_MODEL_TEMPERATURE", "not-a-number");

    let provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: "https://api.test/v1".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "kimi".to_string(),
        model: "kimi-base".to_string(),
        max_context_size: 4096,
        capabilities: None,
    };

    let err = match create_llm(&provider, &model, None, None).await {
        Ok(_) => panic!("expected temperature parsing error"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("could not convert string to float")
    );
}

#[tokio::test]
async fn test_create_llm_kimi_ignores_optional_backend_env_lookup_failures() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::with_failing_keys(
            HashMap::new(),
            HashSet::from([
                "KIMI_MODEL_TEMPERATURE".to_string(),
                "KIMI_MODEL_TOP_P".to_string(),
                "KIMI_MODEL_MAX_TOKENS".to_string(),
            ]),
        );

        let provider = LLMProvider {
            provider_type: ProviderType::Kimi,
            base_url: "https://kimi.test/v1".to_string(),
            api_key: "kimi-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let model = LLMModel {
            provider: "kimi".to_string(),
            model: "kimi-k2-0905-preview".to_string(),
            max_context_size: 128_000,
            capabilities: None,
        };

        let llm = create_llm(&provider, &model, None, None)
            .await
            .expect("optional backend env lookup failure should be ignored")
            .expect("llm");

        assert!(llm.chat_provider.as_any().is::<Kimi>());
    })
    .await;
}

#[tokio::test]
async fn test_create_llm_vertexai_ignores_optional_backend_env_lookup_failures() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::with_failing_keys(
            HashMap::new(),
            HashSet::from(["VERTEXAI_MODEL_TEMPERATURE".to_string()]),
        );

        let provider = LLMProvider {
            provider_type: ProviderType::Vertexai,
            base_url: "https://vertex.test/v1".to_string(),
            api_key: "vertex-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let model = LLMModel {
            provider: "vertex".to_string(),
            model: "gemini-3-pro-preview".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        let llm = create_llm(&provider, &model, None, None)
            .await
            .expect("optional backend env lookup failure should be ignored")
            .expect("llm");

        assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
        assert_eq!(llm.chat_provider.name(), "vertexai");
    })
    .await;
}

#[tokio::test]
async fn test_create_llm_echo_provider() {
    let provider = LLMProvider {
        provider_type: ProviderType::Echo,
        base_url: "".to_string(),
        api_key: "".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "_echo".to_string(),
        model: "echo".to_string(),
        max_context_size: 1234,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<EchoChatProvider>());
    assert_eq!(llm.max_context_size, 1234);
}

#[tokio::test]
async fn test_create_llm_requires_base_url_for_kimi() {
    let provider = LLMProvider {
        provider_type: ProviderType::Kimi,
        base_url: "".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "kimi".to_string(),
        model: "kimi-base".to_string(),
        max_context_size: 4096,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm");
    assert!(llm.is_none());
}

#[tokio::test]
async fn test_create_llm_openai_legacy_provider() {
    let provider = LLMProvider {
        provider_type: ProviderType::OpenaiLegacy,
        base_url: "https://api.openai.test/v1".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "openai".to_string(),
        model: "gpt-4.1".to_string(),
        max_context_size: 200_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
    assert_eq!(llm.chat_provider.name(), "openai");
}

#[tokio::test]
async fn test_create_llm_openai_responses_provider() {
    let provider = LLMProvider {
        provider_type: ProviderType::OpenaiResponses,
        base_url: "https://api.openai.test/v1".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "openai-responses".to_string(),
        model: "gpt-5-codex".to_string(),
        max_context_size: 200_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<OpenAIResponses>());
    assert_eq!(llm.chat_provider.name(), "openai-responses");
}

#[tokio::test]
async fn test_create_llm_anthropic_provider() {
    let provider = LLMProvider {
        provider_type: ProviderType::Anthropic,
        base_url: "https://api.anthropic.test".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4".to_string(),
        max_context_size: 200_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<Anthropic>());
    assert_eq!(llm.chat_provider.name(), "anthropic");
}

#[tokio::test]
async fn test_create_llm_google_genai_provider_uses_openai_compat() {
    let provider = LLMProvider {
        provider_type: ProviderType::GoogleGenai,
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
        api_key: "test-key".to_string(),
        env: None,
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "google".to_string(),
        model: "gemini-2.5-flash".to_string(),
        max_context_size: 1_000_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
    assert_eq!(llm.chat_provider.name(), "google_genai");
}

#[tokio::test]
async fn test_create_llm_google_genai_uses_gemini_temperature_env() {
    let _lock = ENV_LOCK.lock().await;

    with_current_kaos_scope(async {
        let _guard = BackendEnvKaosGuard::new(HashMap::from([(
            "GEMINI_MODEL_TEMPERATURE".to_string(),
            "0.3".to_string(),
        )]));

        let provider = LLMProvider {
            provider_type: ProviderType::GoogleGenai,
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
            api_key: "test-key".to_string(),
            env: None,
            custom_headers: None,
        };
        let model = LLMModel {
            provider: "google".to_string(),
            model: "gemini-2.5-flash".to_string(),
            max_context_size: 1_000_000,
            capabilities: None,
        };

        let llm = create_llm(&provider, &model, None, None)
            .await
            .expect("create llm")
            .expect("llm");

        let openai = llm
            .chat_provider
            .as_any()
            .downcast_ref::<OpenAILegacy>()
            .expect("openai legacy provider");

        assert_eq!(
            serde_json::Value::Object(openai.model_parameters()),
            json!({
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "temperature": 0.3
            })
        );
    })
    .await;
}

#[tokio::test]
async fn test_create_llm_vertexai_provider_does_not_mutate_process_env_and_uses_openai_compat() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("VERTEX_TEST_PROJECT", "old");

    let provider = LLMProvider {
        provider_type: ProviderType::Vertexai,
        base_url: "https://vertex.test/v1".to_string(),
        api_key: "test-key".to_string(),
        env: Some(HashMap::from([(
            "VERTEX_TEST_PROJECT".to_string(),
            "new-project".to_string(),
        )])),
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "vertex".to_string(),
        model: "gemini-3-pro-preview".to_string(),
        max_context_size: 1_000_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
    assert_eq!(llm.chat_provider.name(), "vertexai");
    assert_eq!(
        std::env::var("VERTEX_TEST_PROJECT").expect("vertex env"),
        "old"
    );
}

#[tokio::test]
async fn test_create_llm_scripted_echo_prefers_provider_env_without_mutating_process_env() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("KIMI_SCRIPTED_ECHO_SCRIPTS", "/tmp/does-not-exist.json");

    let temp_dir = TempDir::new().expect("temp dir");
    let scripts_path = temp_dir.path().join("scripts.json");
    std::fs::write(&scripts_path, r#"["text: from provider env"]"#).expect("write scripts");

    let provider = LLMProvider {
        provider_type: ProviderType::ScriptedEcho,
        base_url: String::new(),
        api_key: String::new(),
        env: Some(HashMap::from([
            (
                "KIMI_SCRIPTED_ECHO_SCRIPTS".to_string(),
                scripts_path.to_string_lossy().to_string(),
            ),
            ("KIMI_SCRIPTED_ECHO_TRACE".to_string(), "false".to_string()),
        ])),
        custom_headers: None,
    };
    let model = LLMModel {
        provider: "_scripted_echo".to_string(),
        model: "scripted_echo".to_string(),
        max_context_size: 10_000,
        capabilities: None,
    };

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");

    assert_eq!(llm.chat_provider.name(), "scripted_echo");
    assert_eq!(
        std::env::var("KIMI_SCRIPTED_ECHO_SCRIPTS").expect("scripted echo env"),
        "/tmp/does-not-exist.json"
    );
}

#[tokio::test]
async fn test_create_llm_scripted_echo_uses_host_env_instead_of_backend_env() {
    let _lock = ENV_LOCK.lock().await;
    let temp_dir = TempDir::new().expect("temp dir");
    let scripts_path = temp_dir.path().join("scripts.json");
    std::fs::write(&scripts_path, r#"["text: from host env"]"#).expect("write scripts");
    let host_path = scripts_path.to_string_lossy().to_string();
    let _guard = EnvGuard::set("KIMI_SCRIPTED_ECHO_SCRIPTS", &host_path);

    with_current_kaos_scope(async {
        let _backend = BackendEnvKaosGuard::new(HashMap::from([(
            "KIMI_SCRIPTED_ECHO_SCRIPTS".to_string(),
            "/remote/does-not-exist.json".to_string(),
        )]));

        let llm = create_llm(
            &LLMProvider {
                provider_type: ProviderType::ScriptedEcho,
                base_url: String::new(),
                api_key: String::new(),
                env: None,
                custom_headers: None,
            },
            &LLMModel {
                provider: "_scripted_echo".to_string(),
                model: "scripted_echo".to_string(),
                max_context_size: 10_000,
                capabilities: None,
            },
            None,
            None,
        )
        .await
        .expect("create llm")
        .expect("llm");

        assert_eq!(llm.chat_provider.name(), "scripted_echo");
    })
    .await;
}

#[tokio::test]
async fn test_create_llm_vertexai_uses_provider_scoped_api_key_env_without_mutating_process_env() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("VERTEXAI_API_KEY", "process-old-key");
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer overlay-key"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let mut provider = LLMProvider {
        provider_type: ProviderType::Vertexai,
        base_url: format!("{}/v1", mock_server.uri()),
        api_key: String::new(),
        env: Some(HashMap::from([(
            "VERTEXAI_API_KEY".to_string(),
            "overlay-key".to_string(),
        )])),
        custom_headers: None,
    };
    let mut model = LLMModel {
        provider: "vertex".to_string(),
        model: "gemini-3-pro-preview".to_string(),
        max_context_size: 1_000_000,
        capabilities: None,
    };

    augment_provider_with_env_vars(&mut provider, &mut model)
        .await
        .expect("env overrides");

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");
    assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
    assert_eq!(llm.chat_provider.name(), "vertexai");

    let tools: Vec<kosong::tooling::Tool> = vec![];
    let history: Vec<kosong::message::Message> = vec![];
    let _stream = llm
        .chat_provider
        .generate("", &tools, &history)
        .await
        .expect("generate request should use provider env api key");

    assert_eq!(
        std::env::var("VERTEXAI_API_KEY").expect("vertex env"),
        "process-old-key"
    );
}

#[tokio::test]
async fn test_create_llm_vertexai_explicit_empty_provider_env_api_key_disables_process_env_fallback()
 {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("VERTEXAI_API_KEY", "process-old-key");
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer process-old-key"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
        .expect(0)
        .mount(&mock_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/event-stream"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let mut provider = LLMProvider {
        provider_type: ProviderType::Vertexai,
        base_url: format!("{}/v1", mock_server.uri()),
        api_key: String::new(),
        env: Some(HashMap::from([(
            "VERTEXAI_API_KEY".to_string(),
            String::new(),
        )])),
        custom_headers: None,
    };
    let mut model = LLMModel {
        provider: "vertex".to_string(),
        model: "gemini-3-pro-preview".to_string(),
        max_context_size: 1_000_000,
        capabilities: None,
    };

    augment_provider_with_env_vars(&mut provider, &mut model)
        .await
        .expect("env overrides");

    let llm = create_llm(&provider, &model, None, None)
        .await
        .expect("create llm")
        .expect("llm");
    assert!(llm.chat_provider.as_any().is::<OpenAILegacy>());
    assert_eq!(llm.chat_provider.name(), "vertexai");

    let tools: Vec<kosong::tooling::Tool> = vec![];
    let history: Vec<kosong::message::Message> = vec![];
    let _stream = llm
        .chat_provider
        .generate("", &tools, &history)
        .await
        .expect("generate request should not fall back to process vertex key");

    assert_eq!(
        std::env::var("VERTEXAI_API_KEY").expect("vertex env"),
        "process-old-key"
    );
}
