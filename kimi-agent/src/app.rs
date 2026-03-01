use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use kaos;
use kaos::KaosPath;
use tracing::info;

use crate::agentspec::default_agent_file;
use crate::config::{Config, LLMModel, LLMProvider, ProviderType, load_config};
use crate::llm::{augment_provider_with_env_vars, create_llm};
use crate::session::Session;
use crate::soul::agent::{Runtime, load_agent};
use crate::soul::context::Context;
use crate::soul::kimisoul::KimiSoul;
use crate::soul::run_soul;
use crate::storage::Storage;
use crate::wire::server::{WireServer, WireWsServer, WsSessionRuntimeOptions};
use crate::wire::{WireMessage, WireRecordTarget};

const DEFAULT_FALLBACK_MAX_CONTEXT_SIZE: i64 = 100_000;

pub struct KimiCLI {
    soul: Arc<KimiSoul>,
    runtime: Runtime,
    #[allow(dead_code)]
    env_overrides: HashMap<String, String>,
}

pub enum ConfigInput {
    Path(std::path::PathBuf),
    Inline(Box<Config>),
}

impl ConfigInput {
    async fn load(self) -> Result<Config, crate::exception::ConfigError> {
        match self {
            ConfigInput::Path(path) => load_config(Some(path.as_path())).await,
            ConfigInput::Inline(config) => Ok(*config),
        }
    }
}

pub struct CreateOptions {
    pub config: Option<ConfigInput>,
    pub storage: Option<Storage>,
    pub model_name: Option<String>,
    pub thinking: Option<bool>,
    pub yolo: bool,
    pub agent_file: Option<PathBuf>,
    pub mcp_configs: Vec<serde_json::Value>,
    pub skills_dir: Option<KaosPath>,
    pub max_steps_per_turn: Option<i64>,
    pub max_retries_per_step: Option<i64>,
    pub max_ralph_iterations: Option<i64>,
}

impl KimiCLI {
    pub async fn create(session: Session, options: CreateOptions) -> anyhow::Result<KimiCLI> {
        let CreateOptions {
            config,
            storage,
            model_name,
            thinking,
            yolo,
            agent_file,
            mcp_configs,
            skills_dir,
            max_steps_per_turn,
            max_retries_per_step,
            max_ralph_iterations,
        } = options;

        let mut config = match config {
            Some(config) => config.load().await?,
            None => load_config(None).await?,
        };
        let storage = match storage {
            Some(storage) => storage,
            None => Storage::open(&config.storage).await?,
        };
        if let Some(max_steps) = max_steps_per_turn {
            config.loop_control.max_steps_per_turn = max_steps;
        }
        if let Some(max_retries) = max_retries_per_step {
            config.loop_control.max_retries_per_step = max_retries;
        }
        if let Some(max_ralph) = max_ralph_iterations {
            config.loop_control.max_ralph_iterations = max_ralph;
        }
        info!(
            default_model = %config.default_model,
            models = config.models.len(),
            providers = config.providers.len(),
            "Loaded config"
        );

        let mut model = None;
        let mut provider = None;

        if model_name.is_none()
            && !config.default_model.is_empty()
            && let Some(m) = config.models.get(&config.default_model)
        {
            model = Some(m.clone());
            provider = config.providers.get(&m.provider).cloned();
        }
        if let Some(name) = model_name.as_deref()
            && let Some(m) = config.models.get(name)
        {
            model = Some(m.clone());
            provider = config.providers.get(&m.provider).cloned();
        }

        if model.is_none() {
            model = Some(LLMModel {
                provider: "".to_string(),
                model: "".to_string(),
                max_context_size: 0,
                capabilities: None,
            });
            provider = Some(LLMProvider {
                provider_type: ProviderType::Kimi,
                base_url: "".to_string(),
                api_key: "".to_string(),
                env: None,
                custom_headers: None,
            });
        }

        let mut model = model.unwrap();
        let mut provider = provider.unwrap();

        info!(
            provider_type = ?provider.provider_type,
            base_url = %provider.base_url,
            "Using LLM provider"
        );
        info!(
            model = %model.model,
            max_context_size = model.max_context_size,
            "Using LLM model"
        );
        let env_overrides = augment_provider_with_env_vars(&mut provider, &mut model)
            .await
            .map_err(anyhow::Error::new)?;
        if model.max_context_size <= 0 {
            model.max_context_size = DEFAULT_FALLBACK_MAX_CONTEXT_SIZE;
        }

        let thinking = thinking.unwrap_or(config.default_thinking);
        info!(thinking, "Thinking mode");
        let llm = create_llm(&provider, &model, Some(thinking), Some(&session.id))
            .await
            .map_err(anyhow::Error::new)?
            .map(Arc::new);

        let runtime =
            Runtime::create(config, storage.clone(), llm, session, yolo, skills_dir).await;

        let agent_file = agent_file.unwrap_or_else(default_agent_file);
        let agent = load_agent(&agent_file, runtime.clone(), &mcp_configs).await?;

        let mut context = Context::new(storage, runtime.session.id.clone());
        let _ = context.restore().await;
        let soul = Arc::new(KimiSoul::new(agent, context));

        Ok(KimiCLI {
            soul,
            runtime,
            env_overrides,
        })
    }

    pub fn soul(&self) -> Arc<KimiSoul> {
        Arc::clone(&self.soul)
    }

    pub fn session(&self) -> &Session {
        &self.runtime.session
    }

    pub async fn run(
        &self,
        user_input: crate::wire::UserInput,
        cancel_token: CancellationToken,
        merge_wire_messages: bool,
    ) -> anyhow::Result<Vec<WireMessage>> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let tx_for_ui = tx.clone();
        let work_dir = self.runtime.session.work_dir.clone();
        let original_cwd = KaosPath::cwd();
        let wire_target = WireRecordTarget::new(
            self.runtime.storage.clone(),
            self.runtime.session.id.clone(),
        );

        let ui_loop = move |wire: Arc<crate::wire::Wire>| {
            let tx = tx_for_ui.clone();
            async move {
                let ui = wire.ui_side(merge_wire_messages);
                while let Ok(msg) = ui.receive().await {
                    let _ = tx.send(msg);
                }
                Ok(())
            }
        };

        let _ = kaos::chdir(&work_dir).await;
        let result = run_soul(
            self.soul.as_ref(),
            user_input,
            ui_loop,
            cancel_token,
            Some(wire_target),
        )
        .await;
        let _ = kaos::chdir(&original_cwd).await;

        drop(tx);
        let mut messages = Vec::new();
        while let Some(msg) = rx.recv().await {
            messages.push(msg);
        }

        result?;
        Ok(messages)
    }

    pub async fn run_wire_stdio(&self) -> anyhow::Result<()> {
        let server = WireServer::new(Arc::clone(&self.soul));
        server.serve().await
    }

    pub async fn run_wire_ws(
        &self,
        options: WsSessionRuntimeOptions,
        listen_addr: SocketAddr,
        path: &str,
    ) -> anyhow::Result<()> {
        let server = WireWsServer::new(options, listen_addr, path)?;
        server.serve().await
    }
}
