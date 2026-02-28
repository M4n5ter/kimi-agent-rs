use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde_json::{Map, Value};
use thiserror::Error;

use kosong::chat_provider::{ChatProvider, ChatProviderError, ThinkingEffort};

use crate::config::{LLMModel, LLMProvider, ModelCapability, ProviderType};
use crate::constant::user_agent;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("chat provider error: {0}")]
    ChatProvider(String),
    #[error("scripted echo error: {0}")]
    ScriptedEcho(String),
    #[error("{0}")]
    EnvVar(String),
}

pub struct LLM {
    pub chat_provider: Box<dyn ChatProvider>,
    pub max_context_size: i64,
    pub capabilities: HashSet<ModelCapability>,
    pub model_config: Option<LLMModel>,
    pub provider_config: Option<LLMProvider>,
}

impl LLM {
    pub fn model_name(&self) -> &str {
        self.chat_provider.model_name()
    }
}

#[derive(Clone, Copy)]
struct ProviderCredentialEnvKeys {
    base_url: &'static [&'static str],
    api_key: &'static [&'static str],
}

struct ResolvedProviderCredentials {
    base_url: Option<(&'static str, String)>,
    api_key: Option<(&'static str, String)>,
}

#[derive(Clone, Copy)]
struct OpenAiCompatEnvProfile {
    provider_name: &'static str,
    credentials: ProviderCredentialEnvKeys,
    temperature: &'static [&'static str],
}

pub async fn augment_provider_with_env_vars(
    provider: &mut LLMProvider,
    model: &mut LLMModel,
) -> Result<HashMap<String, String>, LLMError> {
    let mut applied = HashMap::new();
    let resolved_credentials = if let Some(env_keys) =
        provider_credential_env_keys(&provider.provider_type)
    {
        Some(resolve_provider_credentials_from_env_sources(provider.env.as_ref(), env_keys).await?)
    } else {
        None
    };

    if let Some(credentials) = resolved_credentials.as_ref() {
        apply_resolved_provider_credentials(provider, credentials);
        if provider.provider_type == ProviderType::Kimi {
            if let Some((key, value)) = credentials.base_url.as_ref() {
                applied.insert((*key).to_string(), value.clone());
            }
            if let Some((key, _)) = credentials.api_key.as_ref() {
                applied.insert((*key).to_string(), "******".to_string());
            }
        }
    }

    if provider.provider_type == ProviderType::Kimi {
        if let Some(model_name) = read_backend_env_var("KIMI_MODEL_NAME")
            .await?
            .filter(|value| !value.is_empty())
        {
            model.model = model_name.clone();
            applied.insert("KIMI_MODEL_NAME".to_string(), model_name);
        }
        if let Some(max_context_size) = read_backend_env_var("KIMI_MODEL_MAX_CONTEXT_SIZE")
            .await?
            .filter(|value| !value.is_empty())
        {
            let value = parse_env_i64(&max_context_size)?;
            model.max_context_size = value;
            applied.insert("KIMI_MODEL_MAX_CONTEXT_SIZE".to_string(), max_context_size);
        }
        if let Some(caps) = read_backend_env_var("KIMI_MODEL_CAPABILITIES")
            .await?
            .filter(|value| !value.is_empty())
        {
            let mut parsed = HashSet::new();
            for cap in caps.split(',').map(|s| s.trim().to_lowercase()) {
                match cap.as_str() {
                    "image_in" => {
                        parsed.insert(ModelCapability::ImageIn);
                    }
                    "video_in" => {
                        parsed.insert(ModelCapability::VideoIn);
                    }
                    "thinking" => {
                        parsed.insert(ModelCapability::Thinking);
                    }
                    "always_thinking" => {
                        parsed.insert(ModelCapability::AlwaysThinking);
                    }
                    _ => {}
                }
            }
            model.capabilities = Some(parsed);
            applied.insert("KIMI_MODEL_CAPABILITIES".to_string(), caps);
        }
    }

    Ok(applied)
}

pub async fn create_llm(
    provider: &LLMProvider,
    model: &LLMModel,
    thinking: Option<bool>,
    session_id: Option<&str>,
) -> Result<Option<LLM>, LLMError> {
    if provider.provider_type != ProviderType::Echo
        && provider.provider_type != ProviderType::ScriptedEcho
        && (provider.base_url.is_empty() || model.model.is_empty())
    {
        return Ok(None);
    }

    let default_headers = build_provider_headers(provider)?;

    let chat_provider: Box<dyn ChatProvider> = match provider.provider_type {
        ProviderType::Kimi => {
            let mut kimi = kosong::chat_provider::kimi::Kimi::new(
                model.model.clone(),
                Some(provider.api_key.clone()),
                Some(provider.base_url.clone()),
                Some(default_headers.clone()),
            )
            .map_err(map_chat_provider_error)?;

            let mut kwargs = Map::new();
            if let Some(session_id) = session_id {
                kwargs.insert(
                    "prompt_cache_key".to_string(),
                    Value::String(session_id.to_string()),
                );
            }
            if let Some(value) = read_backend_env_var("KIMI_MODEL_TEMPERATURE")
                .await?
                .filter(|value| !value.is_empty())
            {
                let parsed = parse_env_f64(&value)?;
                kwargs.insert("temperature".to_string(), Value::from(parsed));
            }
            if let Some(value) = read_backend_env_var("KIMI_MODEL_TOP_P")
                .await?
                .filter(|value| !value.is_empty())
            {
                let parsed = parse_env_f64(&value)?;
                kwargs.insert("top_p".to_string(), Value::from(parsed));
            }
            if let Some(value) = read_backend_env_var("KIMI_MODEL_MAX_TOKENS")
                .await?
                .filter(|value| !value.is_empty())
            {
                let parsed = parse_env_i64(&value)?;
                kwargs.insert("max_tokens".to_string(), Value::from(parsed));
            }
            if !kwargs.is_empty() {
                kimi = kimi.with_generation_kwargs(kwargs);
            }
            Box::new(kimi)
        }
        ProviderType::OpenaiLegacy => Box::new(
            build_openai_compat_legacy_provider(
                &provider.provider_type,
                provider,
                model,
                default_headers.clone(),
            )
            .await?,
        ),
        ProviderType::OpenaiResponses => {
            let mut openai = kosong::chat_provider::openai_responses::OpenAIResponses::new(
                model.model.clone(),
                Some(provider.api_key.clone()),
                Some(provider.base_url.clone()),
                Some(default_headers.clone()),
            )
            .map_err(map_chat_provider_error)?;
            if let Some(value) = read_non_empty_env_override_var(
                provider.env.as_ref(),
                &["OPENAI_MODEL_TEMPERATURE"],
            )
            .await?
            {
                let parsed = parse_env_f64(&value)?;
                let mut kwargs = Map::new();
                kwargs.insert("temperature".to_string(), Value::from(parsed));
                openai = openai.with_generation_kwargs(kwargs);
            }
            Box::new(openai)
        }
        ProviderType::Anthropic => {
            let mut anthropic = kosong::chat_provider::anthropic::Anthropic::new(
                model.model.clone(),
                Some(provider.api_key.clone()),
                Some(provider.base_url.clone()),
                Some(default_headers.clone()),
            )
            .map_err(map_chat_provider_error)?;

            let mut kwargs = Map::new();
            kwargs.insert("max_tokens".to_string(), Value::from(50_000));
            if let Some(value) = read_backend_env_var("ANTHROPIC_MODEL_TEMPERATURE")
                .await?
                .filter(|value| !value.is_empty())
            {
                kwargs.insert(
                    "temperature".to_string(),
                    Value::from(parse_env_f64(&value)?),
                );
            }
            if let Some(value) = read_backend_env_var("ANTHROPIC_MODEL_TOP_P")
                .await?
                .filter(|value| !value.is_empty())
            {
                kwargs.insert("top_p".to_string(), Value::from(parse_env_f64(&value)?));
            }
            if let Some(value) = read_backend_env_var("ANTHROPIC_MODEL_MAX_TOKENS")
                .await?
                .filter(|value| !value.is_empty())
            {
                kwargs.insert(
                    "max_tokens".to_string(),
                    Value::from(parse_env_i64(&value)?),
                );
            }

            anthropic = anthropic.with_generation_kwargs(kwargs);
            Box::new(anthropic)
        }
        ProviderType::GoogleGenai | ProviderType::Gemini => Box::new(
            build_openai_compat_legacy_provider(
                &provider.provider_type,
                provider,
                model,
                default_headers.clone(),
            )
            .await?,
        ),
        ProviderType::Vertexai => Box::new(
            build_openai_compat_legacy_provider(
                &provider.provider_type,
                provider,
                model,
                default_headers.clone(),
            )
            .await?,
        ),
        ProviderType::Echo => Box::new(kosong::chat_provider::echo::EchoChatProvider),
        ProviderType::ScriptedEcho => {
            let scripts = load_scripted_echo_scripts(provider.env.as_ref()).await?;
            let trace = read_env_var(provider.env.as_ref(), "KIMI_SCRIPTED_ECHO_TRACE")
                .await?
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            let trace_enabled = matches!(trace.as_str(), "1" | "true" | "yes" | "on");
            Box::new(
                kosong::chat_provider::echo::scripted_echo::ScriptedEchoChatProvider::new(
                    scripts,
                    trace_enabled,
                ),
            )
        }
        _ => {
            return Ok(None);
        }
    };

    let capabilities = derive_model_capabilities(model);

    let chat_provider = apply_thinking(chat_provider, &capabilities, thinking);

    Ok(Some(LLM {
        chat_provider,
        max_context_size: model.max_context_size,
        capabilities,
        model_config: Some(model.clone()),
        provider_config: Some(provider.clone()),
    }))
}

pub fn derive_model_capabilities(model: &LLMModel) -> HashSet<ModelCapability> {
    let mut capabilities = model.capabilities.clone().unwrap_or_default();
    let name = model.model.to_lowercase();
    if name.contains("thinking") || name.contains("reason") {
        capabilities.insert(ModelCapability::Thinking);
        capabilities.insert(ModelCapability::AlwaysThinking);
    } else if model.model == "kimi-for-coding" || model.model == "kimi-code" {
        capabilities.insert(ModelCapability::Thinking);
        capabilities.insert(ModelCapability::ImageIn);
        capabilities.insert(ModelCapability::VideoIn);
    }
    capabilities
}

fn apply_thinking(
    chat_provider: Box<dyn ChatProvider>,
    capabilities: &HashSet<ModelCapability>,
    thinking: Option<bool>,
) -> Box<dyn ChatProvider> {
    if capabilities.contains(&ModelCapability::AlwaysThinking)
        || (thinking == Some(true) && capabilities.contains(&ModelCapability::Thinking))
    {
        chat_provider.with_thinking(ThinkingEffort::High)
    } else if thinking == Some(false) {
        chat_provider.with_thinking(ThinkingEffort::Off)
    } else {
        chat_provider
    }
}

fn build_provider_headers(provider: &LLMProvider) -> Result<reqwest::header::HeaderMap, LLMError> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        reqwest::header::HeaderValue::from_str(&user_agent())
            .map_err(|err| LLMError::ChatProvider(err.to_string()))?,
    );
    if let Some(custom) = &provider.custom_headers {
        for (key, value) in custom {
            if let (Ok(header_name), Ok(header_value)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                headers.insert(header_name, header_value);
            }
        }
    }
    Ok(headers)
}

fn parse_env_i64(value: &str) -> Result<i64, LLMError> {
    value.parse::<i64>().map_err(|_| {
        LLMError::EnvVar(format!(
            "invalid literal for int() with base 10: '{}'",
            value
        ))
    })
}

fn parse_env_f64(value: &str) -> Result<f64, LLMError> {
    value
        .parse::<f64>()
        .map_err(|_| LLMError::EnvVar(format!("could not convert string to float: '{}'", value)))
}

async fn read_backend_env_var(key: &str) -> Result<Option<String>, LLMError> {
    // Provider/model env overrides are intentionally scoped to the active backend.
    // This keeps LLM configuration aligned with the selected kaos environment
    // instead of the launcher process environment. Host-local bootstrap
    // exceptions such as KIMI_SHARE_DIR live outside this module.
    kaos::env_var(key).await.map_err(|err| {
        LLMError::EnvVar(format!(
            "failed to read environment variable `{key}`: {err}"
        ))
    })
}

async fn read_env_override_var(
    provider_env: Option<&HashMap<String, String>>,
    keys: &'static [&'static str],
) -> Result<Option<(&'static str, String)>, LLMError> {
    for key in keys {
        if let Some(value) = provider_env.and_then(|envs| envs.get(*key)).cloned() {
            return Ok(Some((*key, value)));
        }
        if let Some(value) = read_backend_env_var(key).await?
            && !value.is_empty()
        {
            return Ok(Some((*key, value)));
        }
    }
    Ok(None)
}

async fn read_non_empty_env_override_var(
    provider_env: Option<&HashMap<String, String>>,
    keys: &'static [&'static str],
) -> Result<Option<String>, LLMError> {
    Ok(read_env_override_var(provider_env, keys)
        .await?
        .map(|(_, value)| value)
        .filter(|value| !value.is_empty()))
}

fn provider_credential_env_keys(provider_type: &ProviderType) -> Option<ProviderCredentialEnvKeys> {
    match provider_type {
        ProviderType::Kimi => Some(ProviderCredentialEnvKeys {
            base_url: &["KIMI_BASE_URL"],
            api_key: &["KIMI_API_KEY"],
        }),
        ProviderType::Anthropic => Some(ProviderCredentialEnvKeys {
            base_url: &["ANTHROPIC_BASE_URL"],
            api_key: &["ANTHROPIC_API_KEY"],
        }),
        _ => openai_compat_env_profile(provider_type).map(|profile| profile.credentials),
    }
}

fn openai_compat_env_profile(provider_type: &ProviderType) -> Option<OpenAiCompatEnvProfile> {
    match provider_type {
        ProviderType::OpenaiLegacy => Some(OpenAiCompatEnvProfile {
            provider_name: "openai",
            credentials: ProviderCredentialEnvKeys {
                base_url: &["OPENAI_BASE_URL"],
                api_key: &["OPENAI_API_KEY"],
            },
            temperature: &["OPENAI_MODEL_TEMPERATURE"],
        }),
        ProviderType::GoogleGenai | ProviderType::Gemini => Some(OpenAiCompatEnvProfile {
            provider_name: "google_genai",
            credentials: ProviderCredentialEnvKeys {
                base_url: &["GEMINI_BASE_URL", "GOOGLE_GENAI_BASE_URL"],
                api_key: &["GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"],
            },
            temperature: &["GEMINI_MODEL_TEMPERATURE", "GOOGLE_GENAI_MODEL_TEMPERATURE"],
        }),
        ProviderType::Vertexai => Some(OpenAiCompatEnvProfile {
            provider_name: "vertexai",
            credentials: ProviderCredentialEnvKeys {
                base_url: &["VERTEXAI_BASE_URL"],
                api_key: &["VERTEXAI_API_KEY"],
            },
            temperature: &["VERTEXAI_MODEL_TEMPERATURE"],
        }),
        _ => None,
    }
}

async fn resolve_provider_credentials_from_env_sources(
    provider_env: Option<&HashMap<String, String>>,
    env_keys: ProviderCredentialEnvKeys,
) -> Result<ResolvedProviderCredentials, LLMError> {
    Ok(ResolvedProviderCredentials {
        base_url: read_env_override_var(provider_env, env_keys.base_url).await?,
        api_key: read_env_override_var(provider_env, env_keys.api_key).await?,
    })
}

fn apply_resolved_provider_credentials(
    provider: &mut LLMProvider,
    credentials: &ResolvedProviderCredentials,
) {
    if let Some((_, base_url)) = credentials.base_url.as_ref() {
        provider.base_url = base_url.clone();
    }
    if let Some((_, api_key)) = credentials.api_key.as_ref() {
        provider.api_key = api_key.clone();
    }
}

async fn build_openai_compat_legacy_provider(
    provider_type: &ProviderType,
    provider: &LLMProvider,
    model: &LLMModel,
    default_headers: reqwest::header::HeaderMap,
) -> Result<kosong::chat_provider::openai_legacy::OpenAILegacy, LLMError> {
    let env_profile = openai_compat_env_profile(provider_type).ok_or_else(|| {
        LLMError::ChatProvider("missing OpenAI-compatible env profile".to_string())
    })?;

    // The transport is OpenAI-compatible, but the env family stays bound to
    // the logical provider type (OpenAI, Gemini, Vertex AI).
    let mut openai = kosong::chat_provider::openai_legacy::OpenAILegacy::new(
        model.model.clone(),
        non_empty_provider_value(&provider.api_key),
        Some(provider.base_url.clone()),
        Some(default_headers),
    )
    .map_err(map_chat_provider_error)?
    .with_provider_name(env_profile.provider_name);

    if let Some(value) =
        read_non_empty_env_override_var(provider.env.as_ref(), env_profile.temperature).await?
    {
        let parsed = parse_env_f64(&value)?;
        let mut kwargs = Map::new();
        kwargs.insert("temperature".to_string(), Value::from(parsed));
        openai = openai.with_generation_kwargs(kwargs);
    }

    Ok(openai)
}

async fn read_env_var(
    provider_env: Option<&HashMap<String, String>>,
    key: &str,
) -> Result<Option<String>, LLMError> {
    if let Some(value) = provider_env.and_then(|envs| envs.get(key)).cloned() {
        return Ok(Some(value));
    }
    read_backend_env_var(key).await
}

fn non_empty_provider_value(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

async fn load_scripted_echo_scripts(
    provider_env: Option<&HashMap<String, String>>,
) -> Result<Vec<String>, LLMError> {
    let script_path = read_env_var(provider_env, "KIMI_SCRIPTED_ECHO_SCRIPTS")
        .await?
        .ok_or_else(|| {
            LLMError::ScriptedEcho(
                "KIMI_SCRIPTED_ECHO_SCRIPTS is required for _scripted_echo.".to_string(),
            )
        })?;
    let path = PathBuf::from(script_path).expanduser();
    if tokio::fs::metadata(&path).await.is_err() {
        return Err(LLMError::ScriptedEcho(format!(
            "Scripted echo file not found: {}",
            path.display()
        )));
    }
    let text = tokio::fs::read_to_string(&path)
        .await
        .map_err(|err| LLMError::ScriptedEcho(err.to_string()))?;
    if let Ok(value) = serde_json::from_str::<Value>(&text) {
        if let Value::Array(items) = value
            && items.iter().all(|item| matches!(item, Value::String(_)))
        {
            return Ok(items
                .into_iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect());
        }
        return Err(LLMError::ScriptedEcho(
            "Scripted echo JSON must be an array of strings.".to_string(),
        ));
    }
    let scripts: Vec<String> = text
        .split("\n---\n")
        .map(|chunk| chunk.trim())
        .filter(|chunk| !chunk.is_empty())
        .map(|chunk| chunk.to_string())
        .collect();
    if scripts.is_empty() {
        return Err(LLMError::ScriptedEcho(
            "Scripted echo file must be a JSON array of strings or a text file split by '\\n---\\n'."
                .to_string(),
        ));
    }
    Ok(scripts)
}

fn map_chat_provider_error(err: ChatProviderError) -> LLMError {
    LLMError::ChatProvider(err.to_string())
}

trait ExpandUser {
    fn expanduser(&self) -> PathBuf;
}

impl ExpandUser for PathBuf {
    fn expanduser(&self) -> PathBuf {
        let Some(home) = dirs::home_dir() else {
            return self.clone();
        };
        let path_str = self.to_string_lossy();
        if path_str == "~" {
            return home;
        }
        if let Some(stripped) = path_str.strip_prefix("~/") {
            return home.join(stripped);
        }
        self.clone()
    }
}
