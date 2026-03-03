use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use kaos::KaosPath;

use crate::agentspec::{default_agent_file, okabe_agent_file};
use crate::app::{ConfigInput, CreateOptions, KimiCLI};
use crate::config::{Config, KaosConfig, load_config, load_config_from_string};
use crate::constant::VERSION;
use crate::session::Session;
use crate::session_id::normalize_session_id;
use crate::storage::Storage;
use crate::utils::init_logging;
use tracing::info;

pub mod info;
pub mod mcp;

struct ResolvedSession {
    session: Session,
    created_new_session: bool,
}

#[derive(Parser, Debug)]
#[command(
    name = "kimi-agent",
    about = "Kimi Agent, the Rust agent server.",
    disable_version_flag = true,
    help_expected = true,
    max_term_width = 100
)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long = "version", short = 'V', help = "Show version and exit.")]
    version: bool,

    #[arg(long, help = "Print verbose information. Default: no.")]
    verbose: bool,

    #[arg(long, help = "Log debug information. Default: no.")]
    debug: bool,

    #[arg(
        long = "work-dir",
        short = 'w',
        value_name = "PATH",
        help = "Working directory for the agent. Default: current directory."
    )]
    work_dir: Option<String>,

    #[arg(
        long = "session",
        short = 'S',
        value_name = "SESSION_ID",
        help = "Session ID to resume for the working directory. Default: new session."
    )]
    session_id: Option<String>,

    #[arg(
        long = "continue",
        short = 'C',
        help = "Continue the previous session for the working directory. Default: no."
    )]
    continue_session: bool,

    #[arg(
        long = "config",
        value_name = "TOML_OR_JSON",
        help = "Config TOML/JSON string to load. Default: none."
    )]
    config_string: Option<String>,

    #[arg(
        long = "config-file",
        value_name = "PATH",
        help = "Config TOML/JSON file to load. Default: ~/.kimi/config.toml."
    )]
    config_file: Option<PathBuf>,

    #[arg(
        long = "model",
        short = 'm',
        help = "LLM model to use. Default: default model set in config file."
    )]
    model_name: Option<String>,

    #[arg(
        long = "thinking",
        help = "Enable thinking mode. Default: default thinking mode set in config file."
    )]
    thinking: bool,

    #[arg(
        long = "no-thinking",
        help = "Enable thinking mode. Default: default thinking mode set in config file."
    )]
    no_thinking: bool,

    #[arg(
        long = "yolo",
        short = 'y',
        visible_aliases = ["yes", "auto-approve"],
        help = "Automatically approve all actions. Default: no.",
    )]
    yolo: bool,

    #[arg(long = "wire", hide = true, help = "Deprecated no-op flag.")]
    wire: bool,

    #[arg(
        long = "wire-transport",
        value_enum,
        default_value = "stdio",
        help = "Wire transport to use. Default: stdio."
    )]
    wire_transport: WireTransport,

    #[arg(
        long = "wire-listen",
        value_name = "ADDR",
        default_value = "127.0.0.1:8765",
        help = "Listen address for websocket wire transport. Used only when --wire-transport ws."
    )]
    wire_listen: String,

    #[arg(
        long = "wire-path",
        value_name = "PATH",
        default_value = "/wire",
        help = "HTTP path for websocket wire transport. Used only when --wire-transport ws."
    )]
    wire_path: String,

    #[arg(
        long = "agent",
        value_enum,
        help = "Builtin agent specification to use. Default: builtin default agent."
    )]
    agent: Option<AgentKind>,

    #[arg(
        long = "agent-file",
        value_name = "PATH",
        help = "Custom agent specification file. Default: builtin default agent."
    )]
    agent_file: Option<PathBuf>,

    #[arg(
        long = "mcp-config-file",
        value_name = "PATH",
        help = "MCP config file to load. Add this option multiple times to specify multiple MCP configs. Default: none."
    )]
    mcp_config_file: Vec<PathBuf>,

    #[arg(
        long = "mcp-config",
        value_name = "JSON",
        help = "MCP config JSON to load. Add this option multiple times to specify multiple MCP configs. Default: none."
    )]
    mcp_config: Vec<String>,

    #[arg(
        long = "skills-dir",
        value_name = "PATH",
        help = "Path to the skills directory. Disables builtin/user/project discovery and loads only this directory."
    )]
    skills_dir: Option<String>,

    #[arg(
        long = "max-steps-per-turn",
        value_name = "N",
        help = "Maximum number of steps in one turn. Default: from config."
    )]
    max_steps_per_turn: Option<i64>,

    #[arg(
        long = "max-retries-per-step",
        value_name = "N",
        help = "Maximum number of retries in one step. Default: from config."
    )]
    max_retries_per_step: Option<i64>,

    #[arg(
        long = "max-ralph-iterations",
        value_name = "N",
        help = "Extra iterations after the first turn in Ralph mode. Use -1 for unlimited. Default: from config."
    )]
    max_ralph_iterations: Option<i64>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AgentKind {
    Default,
    Okabe,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum WireTransport {
    Stdio,
    Ws,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show version and protocol information.
    Info(info::InfoArgs),
    /// Manage MCP server configurations.
    Mcp(mcp::McpArgs),
}

struct KaosScopeGuard {
    token: Option<kaos::CurrentKaosToken>,
}

impl KaosScopeGuard {
    fn new(token: kaos::CurrentKaosToken) -> Self {
        Self { token: Some(token) }
    }
}

impl Drop for KaosScopeGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            kaos::reset_current_kaos(token);
        }
    }
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.version {
        println!("kimi-agent, version {VERSION}");
        return Ok(());
    }

    init_logging(cli.debug).await?;
    validate_cli_args(&cli).await?;

    let mcp_runtime_config = if cli
        .command
        .as_ref()
        .is_some_and(|command| matches!(command, Commands::Mcp(_)))
    {
        Some(load_effective_config(&cli).await?)
    } else {
        None
    };

    if let Some(command) = cli.command {
        return match command {
            Commands::Info(args) => {
                info::run_info_command(args);
                Ok(())
            }
            Commands::Mcp(args) => {
                let config = mcp_runtime_config.expect("mcp config should be preloaded");
                let _kaos_guard = init_current_kaos(&config).await?;
                mcp::run_mcp_command(args, &config).await
            }
        };
    }

    let config = load_effective_config(&cli).await?;
    let _kaos_guard = init_current_kaos(&config).await?;
    validate_runtime_paths(&cli).await?;
    let storage = Storage::open(&config.storage).await?;

    let mcp_configs =
        mcp::load_mcp_configs(&storage, &config, &cli.mcp_config_file, &cli.mcp_config).await?;

    let skills_dir = cli
        .skills_dir
        .as_ref()
        .map(|path| cli_path_to_kaos_path(path));

    let work_dir = match cli.work_dir.as_ref() {
        Some(path) => cli_path_to_kaos_path(path),
        None => KaosPath::cwd(),
    };

    let agent_file = resolve_agent_file(cli.agent, cli.agent_file.as_ref())?;
    let thinking = resolve_thinking(cli.thinking, cli.no_thinking)?;

    match cli.wire_transport {
        WireTransport::Stdio => {
            let resolved_session = resolve_session(
                &storage,
                &config.kaos,
                &work_dir,
                cli.session_id.as_ref(),
                cli.continue_session,
            )
            .await?;
            let instance = KimiCLI::create(
                resolved_session.session,
                CreateOptions {
                    config: Some(ConfigInput::Inline(Box::new(config))),
                    storage: Some(storage.clone()),
                    model_name: cli.model_name.clone(),
                    thinking,
                    yolo: cli.yolo,
                    agent_file,
                    mcp_configs,
                    skills_dir,
                    max_steps_per_turn: cli.max_steps_per_turn,
                    max_retries_per_step: cli.max_retries_per_step,
                    max_ralph_iterations: cli.max_ralph_iterations,
                },
            )
            .await?;
            instance
                .run_wire_stdio(resolved_session.created_new_session)
                .await?;
        }
        WireTransport::Ws => {
            let listen_addr = parse_wire_listen_addr(&cli.wire_listen)?;
            let default_session_id = resolve_ws_default_session_id(
                &storage,
                &config.kaos,
                &work_dir,
                cli.session_id.as_ref(),
                cli.continue_session,
            )
            .await?;
            let server = crate::wire::server::WireWsServer::new(
                crate::wire::server::WsSessionRuntimeOptions {
                    storage,
                    work_dir,
                    default_session_id,
                    config,
                    model_name: cli.model_name.clone(),
                    thinking,
                    yolo: cli.yolo,
                    agent_file,
                    mcp_configs,
                    skills_dir,
                    max_steps_per_turn: cli.max_steps_per_turn,
                    max_retries_per_step: cli.max_retries_per_step,
                    max_ralph_iterations: cli.max_ralph_iterations,
                },
                listen_addr,
                &cli.wire_path,
            )?;
            server.serve().await?;
        }
    }
    Ok(())
}

async fn load_effective_config(cli: &Cli) -> Result<Config> {
    if let Some(config_string) = cli.config_string.as_ref() {
        return load_config_from_string(config_string).map_err(anyhow::Error::new);
    }
    if let Some(config_file) = cli.config_file.as_ref() {
        return load_config(Some(config_file))
            .await
            .map_err(anyhow::Error::new);
    }
    load_config(None).await.map_err(anyhow::Error::new)
}

async fn init_current_kaos(config: &Config) -> Result<KaosScopeGuard> {
    let kaos: Arc<dyn kaos::Kaos> = match &config.kaos {
        KaosConfig::Local => Arc::new(kaos::LocalKaos::new()),
        KaosConfig::Ssh { options } => Arc::new(
            kaos::SshKaos::connect(options.clone())
                .await
                .with_context(|| {
                    format!(
                        "Failed to connect ssh kaos {}:{}",
                        options.host, options.port
                    )
                })?,
        ),
    };
    Ok(KaosScopeGuard::new(kaos::set_current_kaos(kaos)))
}

fn cli_path_to_kaos_path(path: &str) -> KaosPath {
    KaosPath::new(path)
}

async fn validate_runtime_paths(cli: &Cli) -> Result<()> {
    if let Some(work_dir) = cli.work_dir.as_ref() {
        ensure_kaos_dir_exists(&cli_path_to_kaos_path(work_dir), "work dir").await?;
    }

    if let Some(skills_dir) = cli.skills_dir.as_ref() {
        ensure_kaos_dir_exists(&cli_path_to_kaos_path(skills_dir), "skills dir").await?;
    }

    Ok(())
}

async fn ensure_kaos_dir_exists(path: &KaosPath, label: &str) -> Result<()> {
    if !path.exists(true).await {
        anyhow::bail!("{label} does not exist: {}", path.to_string_lossy());
    }
    if !path.is_dir(true).await {
        anyhow::bail!("{label} is not a directory: {}", path.to_string_lossy());
    }
    Ok(())
}

async fn validate_cli_args(cli: &Cli) -> Result<()> {
    let conflict_sets = vec![
        vec![
            ("--agent", cli.agent.is_some()),
            ("--agent-file", cli.agent_file.is_some()),
        ],
        vec![
            ("--continue", cli.continue_session),
            ("--session", cli.session_id.is_some()),
        ],
        vec![
            ("--config", cli.config_string.is_some()),
            ("--config-file", cli.config_file.is_some()),
        ],
    ];

    for option_set in conflict_sets {
        let active: Vec<&str> = option_set
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(flag, _)| *flag)
            .collect();
        if active.len() > 1 {
            anyhow::bail!("Cannot combine {}.", active.join(", "));
        }
    }

    if cli.thinking && cli.no_thinking {
        anyhow::bail!("Cannot combine --thinking and --no-thinking.");
    }

    if let Some(session_id) = cli.session_id.as_ref()
        && session_id.trim().is_empty()
    {
        anyhow::bail!("Session ID cannot be empty.");
    }

    if let Some(config_string) = cli.config_string.as_ref()
        && config_string.trim().is_empty()
    {
        anyhow::bail!("Config cannot be empty.");
    }

    if let Some(config_file) = cli.config_file.as_ref() {
        ensure_file_exists(config_file, "config file").await?;
    }

    if let Some(agent_file) = cli.agent_file.as_ref() {
        ensure_file_exists(agent_file, "agent file").await?;
    }

    if let Some(max_steps) = cli.max_steps_per_turn
        && max_steps < 1
    {
        anyhow::bail!("max-steps-per-turn must be >= 1.");
    }

    if let Some(max_retries) = cli.max_retries_per_step
        && max_retries < 1
    {
        anyhow::bail!("max-retries-per-step must be >= 1.");
    }

    if let Some(max_ralph) = cli.max_ralph_iterations
        && max_ralph < -1
    {
        anyhow::bail!("max-ralph-iterations must be >= -1.");
    }

    if matches!(cli.wire_transport, WireTransport::Ws) {
        parse_wire_listen_addr(&cli.wire_listen)?;
        validate_wire_path(&cli.wire_path)?;
    }

    Ok(())
}

async fn ensure_file_exists(path: &Path, label: &str) -> Result<()> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("{label} does not exist: {}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("{label} is not a file: {}", path.display());
    }
    Ok(())
}

fn resolve_agent_file(
    agent: Option<AgentKind>,
    agent_file: Option<&PathBuf>,
) -> Result<Option<PathBuf>> {
    if let Some(agent_file) = agent_file {
        return Ok(Some(agent_file.clone()));
    }
    match agent {
        Some(AgentKind::Default) => Ok(Some(default_agent_file())),
        Some(AgentKind::Okabe) => Ok(Some(okabe_agent_file())),
        None => Ok(None),
    }
}

fn resolve_thinking(thinking: bool, no_thinking: bool) -> Result<Option<bool>> {
    if thinking && no_thinking {
        anyhow::bail!("Cannot combine --thinking and --no-thinking.");
    }
    if thinking {
        Ok(Some(true))
    } else if no_thinking {
        Ok(Some(false))
    } else {
        Ok(None)
    }
}

fn parse_wire_listen_addr(addr: &str) -> Result<SocketAddr> {
    addr.parse()
        .with_context(|| format!("Invalid --wire-listen address: {addr}"))
}

fn validate_wire_path(path: &str) -> Result<()> {
    let path = path.trim();
    if path.is_empty() {
        anyhow::bail!("--wire-path cannot be empty.");
    }
    if !path.starts_with('/') {
        anyhow::bail!("--wire-path must start with '/'.");
    }
    if path.contains('?') || path.contains('#') {
        anyhow::bail!("--wire-path cannot contain query or fragment.");
    }
    Ok(())
}

async fn resolve_session(
    storage: &Storage,
    kaos: &KaosConfig,
    work_dir: &KaosPath,
    session_id: Option<&String>,
    continue_session: bool,
) -> Result<ResolvedSession> {
    if let Some(session_id) = session_id {
        let normalized = normalize_session_id(session_id)
            .map_err(|err| anyhow::anyhow!("Invalid --session value: {err}"))?;
        let found =
            Session::find(storage.clone(), kaos.clone(), work_dir.clone(), &normalized).await?;
        if let Some(session) = found {
            info!("Switching to session: {}", session.id);
            return Ok(ResolvedSession {
                session,
                created_new_session: false,
            });
        }
        info!("Session {} not found, creating new session", normalized);
        let session = Session::create(
            storage.clone(),
            kaos.clone(),
            work_dir.clone(),
            Some(normalized),
        )
        .await?;
        info!("Switching to session: {}", session.id);
        return Ok(ResolvedSession {
            session,
            created_new_session: true,
        });
    }

    if continue_session {
        if let Some(session) =
            Session::continue_(storage.clone(), kaos.clone(), work_dir.clone()).await?
        {
            info!("Continuing previous session: {}", session.id);
            return Ok(ResolvedSession {
                session,
                created_new_session: false,
            });
        }
        anyhow::bail!("No previous session found for the working directory.");
    }

    let session = Session::create(storage.clone(), kaos.clone(), work_dir.clone(), None).await?;
    info!("Created new session: {}", session.id);
    Ok(ResolvedSession {
        session,
        created_new_session: true,
    })
}

async fn resolve_ws_default_session_id(
    storage: &Storage,
    kaos: &KaosConfig,
    work_dir: &KaosPath,
    session_id: Option<&String>,
    continue_session: bool,
) -> Result<String> {
    if let Some(session_id) = session_id {
        return normalize_session_id(session_id)
            .map_err(|err| anyhow::anyhow!("Invalid --session value: {err}"));
    }

    if continue_session {
        if let Some(session) =
            Session::continue_(storage.clone(), kaos.clone(), work_dir.clone()).await?
        {
            return normalize_session_id(&session.id).map_err(|err| {
                anyhow::anyhow!("Invalid session ID in previous session metadata: {err}")
            });
        }
        anyhow::bail!("No previous session found for the working directory.");
    }

    Ok(uuid::Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use kaos::KaosPath;

    use super::resolve_ws_default_session_id;
    use crate::config::{KaosConfig, StorageConfig};
    use crate::storage::Storage;

    #[tokio::test]
    async fn ws_default_session_id_rejects_invalid_custom_session() {
        let work_dir = TempDir::new().expect("work dir");
        let work_path = KaosPath::from(work_dir.path().to_path_buf());
        let storage_root = TempDir::new().expect("storage root");
        let storage = Storage::open(&StorageConfig {
            database_path: storage_root.path().join("state.db").display().to_string(),
            busy_timeout_ms: 1_000,
        })
        .await
        .expect("open storage");
        let invalid_session = "bad:id".to_string();

        let result = resolve_ws_default_session_id(
            &storage,
            &KaosConfig::Local,
            &work_path,
            Some(&invalid_session),
            false,
        )
        .await;
        assert!(result.is_err());
    }
}
