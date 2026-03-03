use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::Local;
use kaos::KaosPath;
use kosong::chat_provider::ThinkingEffort;
use regex::Regex;
use tracing::{debug, info};

use crate::agentspec::load_agent_spec;
use crate::config::Config;
use crate::exception::{AgentSpecError, SystemPromptTemplateError};
use crate::llm::{LLM, create_llm};
use crate::session::Session;
use crate::skill::{Skill, discover_skills_from_roots, index_skills, resolve_skills_roots};
use crate::soul::approval::Approval;
use crate::soul::denwarenji::DenwaRenji;
use crate::soul::toolset::{KimiToolset, ToolOverlay};
use crate::storage::Storage;
use crate::utils::{Environment, list_directory};
use serde_json::Value;

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct BuiltinSystemPromptArgs {
    pub KIMI_NOW: String,
    pub KIMI_WORK_DIR: KaosPath,
    pub KIMI_WORK_DIR_LS: String,
    pub KIMI_AGENTS_MD: String,
    pub KIMI_SKILLS: String,
}

impl BuiltinSystemPromptArgs {
    fn as_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("KIMI_NOW".to_string(), self.KIMI_NOW.clone());
        map.insert(
            "KIMI_WORK_DIR".to_string(),
            self.KIMI_WORK_DIR.to_string_lossy(),
        );
        map.insert(
            "KIMI_WORK_DIR_LS".to_string(),
            self.KIMI_WORK_DIR_LS.clone(),
        );
        map.insert("KIMI_AGENTS_MD".to_string(), self.KIMI_AGENTS_MD.clone());
        map.insert("KIMI_SKILLS".to_string(), self.KIMI_SKILLS.clone());
        map
    }
}

pub async fn load_agents_md(work_dir: &KaosPath) -> Option<String> {
    let candidates = [
        work_dir.clone() / "AGENTS.md",
        work_dir.clone() / "agents.md",
    ];
    for path in candidates {
        if path.is_file(true).await
            && let Ok(text) = path.read_text().await
        {
            info!("Loaded agents.md: {}", path.to_string_lossy());
            return Some(text.trim().to_string());
        }
    }
    info!("No AGENTS.md found in {}", work_dir.to_string_lossy());
    None
}

#[derive(Clone)]
pub struct Runtime {
    pub factory: Arc<RuntimeFactory>,
    pub config: Config,
    pub storage: Storage,
    pub llm: Option<Arc<LLM>>,
    pub session: Session,
    pub builtin_args: BuiltinSystemPromptArgs,
    pub denwa_renji: Arc<tokio::sync::Mutex<DenwaRenji>>,
    pub approval: Arc<Approval>,
    pub subagent_registry: Arc<tokio::sync::Mutex<SubagentRegistry>>,
    pub environment: Environment,
    pub skills: HashMap<String, Skill>,
}

#[derive(Clone)]
struct LlmBlueprint {
    provider: crate::config::LLMProvider,
    model: crate::config::LLMModel,
    thinking: Option<bool>,
}

#[derive(Clone)]
pub struct RuntimeFactory {
    config: Config,
    storage: Storage,
    llm: Option<LlmBlueprint>,
    approval: Arc<Approval>,
    environment: Environment,
    skills_dir: Option<KaosPath>,
}

impl RuntimeFactory {
    fn new(
        config: Config,
        storage: Storage,
        llm: Option<Arc<LLM>>,
        approval: Arc<Approval>,
        environment: Environment,
        skills_dir: Option<KaosPath>,
    ) -> Self {
        let llm = llm.and_then(|llm| {
            Some(LlmBlueprint {
                provider: llm.provider_config.clone()?,
                model: llm.model_config.clone()?,
                thinking: llm
                    .chat_provider
                    .thinking_effort()
                    .map(|effort| effort != ThinkingEffort::Off),
            })
        });
        Self {
            config,
            storage,
            llm,
            approval,
            environment,
            skills_dir,
        }
    }

    async fn create_runtime(
        self: &Arc<Self>,
        session: Session,
        subagent_registry: Arc<tokio::sync::Mutex<SubagentRegistry>>,
    ) -> Result<Runtime, anyhow::Error> {
        let work_dir = session.work_dir.clone();
        let (ls_output, agents_md) =
            tokio::join!(list_directory(&work_dir), load_agents_md(&work_dir));

        let skills_roots = resolve_skills_roots(&work_dir, self.skills_dir.clone()).await;
        let skills = discover_skills_from_roots(&skills_roots).await;
        let skills_by_name = index_skills(&skills);
        let skills_formatted = if skills.is_empty() {
            "No skills found.".to_string()
        } else {
            skills
                .iter()
                .map(|skill| {
                    format!(
                        "- {}\n  - Path: {}\n  - Description: {}",
                        skill.name,
                        skill.skill_md_file().to_string_lossy(),
                        skill.description
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let llm = match &self.llm {
            Some(blueprint) => create_llm(
                &blueprint.provider,
                &blueprint.model,
                blueprint.thinking,
                Some(&session.id),
            )
            .await
            .map_err(anyhow::Error::new)?
            .map(Arc::new),
            None => None,
        };

        Ok(Runtime {
            factory: Arc::clone(self),
            config: self.config.clone(),
            storage: self.storage.clone(),
            llm,
            session,
            builtin_args: BuiltinSystemPromptArgs {
                KIMI_NOW: Local::now().to_rfc3339(),
                KIMI_WORK_DIR: work_dir,
                KIMI_WORK_DIR_LS: ls_output,
                KIMI_AGENTS_MD: agents_md.unwrap_or_default(),
                KIMI_SKILLS: skills_formatted,
            },
            denwa_renji: Arc::new(tokio::sync::Mutex::new(DenwaRenji::new())),
            approval: Arc::new(self.approval.share()),
            subagent_registry,
            environment: self.environment.clone(),
            skills: skills_by_name,
        })
    }
}

impl Runtime {
    pub async fn create(
        config: Config,
        storage: Storage,
        llm: Option<Arc<LLM>>,
        session: Session,
        yolo: bool,
        skills_dir: Option<KaosPath>,
    ) -> Runtime {
        let work_dir = session.work_dir.clone();
        let (ls_output, agents_md, environment) = tokio::join!(
            list_directory(&work_dir),
            load_agents_md(&work_dir),
            Environment::detect()
        );

        let skills_roots = resolve_skills_roots(&work_dir, skills_dir.clone()).await;
        let skills = discover_skills_from_roots(&skills_roots).await;
        info!("Discovered {} skill(s)", skills.len());
        let skills_by_name = index_skills(&skills);
        let skills_formatted = if skills.is_empty() {
            "No skills found.".to_string()
        } else {
            skills
                .iter()
                .map(|skill| {
                    format!(
                        "- {}\n  - Path: {}\n  - Description: {}",
                        skill.name,
                        skill.skill_md_file().to_string_lossy(),
                        skill.description
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let approval = Arc::new(Approval::new(yolo));
        let root_llm = llm.clone();
        let factory = Arc::new(RuntimeFactory::new(
            config,
            storage,
            llm,
            Arc::clone(&approval),
            environment.clone(),
            skills_dir,
        ));

        Runtime {
            factory: Arc::clone(&factory),
            config: factory.config.clone(),
            storage: factory.storage.clone(),
            llm: root_llm,
            session,
            builtin_args: BuiltinSystemPromptArgs {
                KIMI_NOW: Local::now().to_rfc3339(),
                KIMI_WORK_DIR: work_dir,
                KIMI_WORK_DIR_LS: ls_output,
                KIMI_AGENTS_MD: agents_md.unwrap_or_default(),
                KIMI_SKILLS: skills_formatted,
            },
            denwa_renji: Arc::new(tokio::sync::Mutex::new(DenwaRenji::new())),
            approval,
            subagent_registry: Arc::new(tokio::sync::Mutex::new(SubagentRegistry::new())),
            environment,
            skills: skills_by_name,
        }
    }

    pub fn copy_for_fixed_subagent(&self) -> Runtime {
        Runtime {
            factory: Arc::clone(&self.factory),
            config: self.config.clone(),
            storage: self.storage.clone(),
            llm: self.llm.clone(),
            session: self.session.clone(),
            builtin_args: self.builtin_args.clone(),
            denwa_renji: Arc::new(tokio::sync::Mutex::new(DenwaRenji::new())),
            approval: Arc::new(self.approval.share()),
            subagent_registry: Arc::clone(&self.subagent_registry),
            environment: self.environment.clone(),
            skills: self.skills.clone(),
        }
    }

    pub async fn create_child_runtime(&self, session: Session) -> Result<Runtime, anyhow::Error> {
        self.factory
            .create_runtime(session, Arc::clone(&self.subagent_registry))
            .await
    }

    pub async fn create_isolated_runtime(
        &self,
        session: Session,
    ) -> Result<Runtime, anyhow::Error> {
        let fixed_subagents = self.subagent_registry.lock().await.fixed_snapshot();
        let mut registry = SubagentRegistry::new();
        registry
            .replace_fixed_subagents(fixed_subagents)
            .map_err(anyhow::Error::msg)?;
        self.factory
            .create_runtime(session, Arc::new(tokio::sync::Mutex::new(registry)))
            .await
    }
}

#[derive(Clone)]
pub struct AgentDefinition {
    pub name: String,
    pub system_prompt: String,
    pub tool_paths: Vec<String>,
    pub mcp_configs: Vec<Value>,
}

impl AgentDefinition {
    pub async fn instantiate(self: &Arc<Self>, runtime: Runtime) -> Result<Agent, anyhow::Error> {
        self.instantiate_with_overlay(runtime, &ToolOverlay::default())
            .await
    }

    pub async fn instantiate_with_overlay(
        self: &Arc<Self>,
        runtime: Runtime,
        overlay: &ToolOverlay,
    ) -> Result<Agent, anyhow::Error> {
        let toolset = Arc::new(tokio::sync::Mutex::new(KimiToolset::new()));
        {
            let mut guard = toolset.lock().await;
            guard
                .load_tools(
                    &self.tool_paths,
                    &runtime,
                    Arc::clone(&toolset),
                    Arc::clone(self),
                )
                .map_err(anyhow::Error::from)?;

            if !self.mcp_configs.is_empty() {
                guard
                    .load_mcp_tools(&self.mcp_configs, &runtime, Arc::clone(&toolset))
                    .await?;
            }
            guard.apply_overlay(overlay).map_err(anyhow::Error::msg)?;
        }

        Ok(Agent {
            name: self.name.clone(),
            system_prompt: self.system_prompt.clone(),
            definition: Arc::clone(self),
            toolset,
            runtime,
        })
    }

    pub fn derive_dynamic(&self, name: String, system_prompt: String) -> Arc<Self> {
        Arc::new(Self {
            name,
            system_prompt,
            tool_paths: self.tool_paths.clone(),
            mcp_configs: self.mcp_configs.clone(),
        })
    }
}

#[derive(Clone)]
pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    pub definition: Arc<AgentDefinition>,
    pub toolset: Arc<tokio::sync::Mutex<KimiToolset>>,
    pub runtime: Runtime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisteredSubagentKind {
    Fixed,
    Dynamic,
}

#[derive(Clone)]
pub struct RegisteredSubagent {
    pub definition: Arc<AgentDefinition>,
    pub description: String,
    pub kind: RegisteredSubagentKind,
}

impl RegisteredSubagent {
    fn fixed(definition: Arc<AgentDefinition>, description: String) -> Self {
        Self {
            definition,
            description,
            kind: RegisteredSubagentKind::Fixed,
        }
    }

    fn dynamic(definition: Arc<AgentDefinition>) -> Self {
        Self {
            definition,
            description: String::new(),
            kind: RegisteredSubagentKind::Dynamic,
        }
    }
}

#[derive(Default)]
pub struct SubagentRegistry {
    fixed_subagents: HashMap<String, RegisteredSubagent>,
    dynamic_subagents: HashMap<String, RegisteredSubagent>,
}

impl SubagentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn replace_fixed_subagents(
        &mut self,
        fixed_subagents: HashMap<String, RegisteredSubagent>,
    ) -> Result<(), String> {
        if let Some(name) = fixed_subagents
            .keys()
            .find(|name| self.dynamic_subagents.contains_key(*name))
        {
            return Err(format!("Duplicate subagent name: {name}"));
        }
        self.fixed_subagents = fixed_subagents;
        Ok(())
    }

    pub fn add_fixed_subagent(
        &mut self,
        name: String,
        definition: Arc<AgentDefinition>,
        description: String,
    ) -> Result<(), String> {
        if self.contains(&name) {
            return Err(format!("Duplicate subagent name: {name}"));
        }
        self.fixed_subagents
            .insert(name, RegisteredSubagent::fixed(definition, description));
        Ok(())
    }

    pub fn add_dynamic_subagent(&mut self, name: String, agent: Arc<AgentDefinition>) {
        self.dynamic_subagents
            .insert(name, RegisteredSubagent::dynamic(agent));
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fixed_subagents.contains_key(name) || self.dynamic_subagents.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<&RegisteredSubagent> {
        self.fixed_subagents
            .get(name)
            .or_else(|| self.dynamic_subagents.get(name))
    }

    pub fn all_dynamic_subagents(&self) -> HashMap<String, Arc<AgentDefinition>> {
        self.dynamic_subagents
            .iter()
            .map(|(name, subagent)| (name.clone(), Arc::clone(&subagent.definition)))
            .collect()
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut names = self
            .fixed_subagents
            .keys()
            .chain(self.dynamic_subagents.keys())
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        names
    }

    pub fn fixed_subagent_descriptions(&self) -> Vec<(String, String)> {
        let mut descriptions = self
            .fixed_subagents
            .iter()
            .map(|(name, subagent)| (name.clone(), subagent.description.clone()))
            .collect::<Vec<_>>();
        descriptions.sort_by(|a, b| a.0.cmp(&b.0));
        descriptions
    }

    pub fn fixed_snapshot(&self) -> HashMap<String, RegisteredSubagent> {
        self.fixed_subagents.clone()
    }
}

struct LoadedAgentDefinition {
    definition: Arc<AgentDefinition>,
    fixed_subagents: HashMap<String, RegisteredSubagent>,
}

fn load_agent_definition<'a>(
    agent_file: &'a Path,
    runtime: Runtime,
    mcp_configs: &'a [serde_json::Value],
) -> futures::future::BoxFuture<'a, Result<LoadedAgentDefinition, anyhow::Error>> {
    Box::pin(async move {
        info!("Loading agent: {}", agent_file.display());
        let agent_spec = load_agent_spec(agent_file).await?;
        let system_prompt = load_system_prompt(
            &agent_spec.system_prompt_path,
            &agent_spec.system_prompt_args,
            &runtime.builtin_args,
        )
        .await?;

        let mut fixed_subagents = HashMap::new();
        for (subagent_name, subagent_spec) in agent_spec.subagents.iter() {
            debug!("Loading subagent: {}", subagent_name);
            let loaded_subagent = load_agent_definition(
                &subagent_spec.path,
                runtime.copy_for_fixed_subagent(),
                mcp_configs,
            )
            .await?;
            register_fixed_subagent(
                &mut fixed_subagents,
                subagent_name,
                RegisteredSubagent::fixed(
                    Arc::clone(&loaded_subagent.definition),
                    subagent_spec.description.clone(),
                ),
            )?;
            for (nested_name, nested_subagent) in loaded_subagent.fixed_subagents {
                register_fixed_subagent(&mut fixed_subagents, &nested_name, nested_subagent)?;
            }
        }

        let mut tools = agent_spec.tools.clone();
        if !agent_spec.exclude_tools.is_empty() {
            debug!("Excluding tools: {:?}", agent_spec.exclude_tools);
            tools.retain(|tool| !agent_spec.exclude_tools.contains(tool));
        }

        Ok(LoadedAgentDefinition {
            definition: Arc::new(AgentDefinition {
                name: agent_spec.name,
                system_prompt,
                tool_paths: tools,
                mcp_configs: mcp_configs.to_vec(),
            }),
            fixed_subagents,
        })
    })
}

fn validate_fixed_subagents<'a>(
    fixed_subagents: &'a HashMap<String, RegisteredSubagent>,
    runtime: Runtime,
) -> futures::future::BoxFuture<'a, Result<(), anyhow::Error>> {
    Box::pin(async move {
        for subagent in fixed_subagents.values() {
            let agent = subagent
                .definition
                .instantiate(runtime.copy_for_fixed_subagent())
                .await?;
            agent.toolset.lock().await.cleanup().await;
        }
        Ok(())
    })
}

pub fn load_agent<'a>(
    agent_file: &'a Path,
    runtime: Runtime,
    mcp_configs: &'a [serde_json::Value],
) -> futures::future::BoxFuture<'a, Result<Agent, anyhow::Error>> {
    Box::pin(async move {
        let loaded = load_agent_definition(agent_file, runtime.clone(), mcp_configs).await?;
        validate_fixed_subagents(&loaded.fixed_subagents, runtime.copy_for_fixed_subagent())
            .await?;
        runtime
            .subagent_registry
            .lock()
            .await
            .replace_fixed_subagents(loaded.fixed_subagents)
            .map_err(anyhow::Error::msg)?;
        loaded.definition.instantiate(runtime).await
    })
}

fn register_fixed_subagent(
    fixed_subagents: &mut HashMap<String, RegisteredSubagent>,
    name: &str,
    subagent: RegisteredSubagent,
) -> Result<(), anyhow::Error> {
    if fixed_subagents.insert(name.to_string(), subagent).is_some() {
        anyhow::bail!("Duplicate fixed subagent name: {name}");
    }
    Ok(())
}

async fn load_system_prompt(
    path: &Path,
    args: &HashMap<String, String>,
    builtin_args: &BuiltinSystemPromptArgs,
) -> Result<String, anyhow::Error> {
    info!("Loading system prompt: {}", path.display());
    let system_prompt = tokio::fs::read_to_string(path).await.map_err(|err| {
        AgentSpecError::new(format!(
            "Failed to read system prompt {}: {err}",
            path.display()
        ))
    })?;

    let mut values = builtin_args.as_map();
    for (key, value) in args {
        values.insert(key.clone(), value.clone());
    }
    debug!(
        "Substituting system prompt with builtin args: {:?}, spec args: {:?}",
        builtin_args.as_map(),
        args
    );

    let rendered = substitute_template(system_prompt.trim(), &values).map_err(|missing| {
        SystemPromptTemplateError::new(format!(
            "Missing system prompt arg in {}: {}",
            path.display(),
            missing.join(", ")
        ))
    })?;
    Ok(rendered)
}

fn substitute_template(
    template: &str,
    values: &HashMap<String, String>,
) -> Result<String, Vec<String>> {
    let re = Regex::new(r"\$\{([A-Za-z0-9_]+)\}").expect("valid system prompt placeholder regex");

    let mut missing: Vec<String> = Vec::new();
    let result = re
        .replace_all(template, |caps: &regex::Captures<'_>| {
            let key = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            match values.get(key) {
                Some(value) => value.clone(),
                None => {
                    missing.push(key.to_string());
                    caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
                }
            }
        })
        .to_string();

    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        return Err(missing);
    }

    Ok(result)
}
