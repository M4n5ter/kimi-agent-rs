use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use boxlite::runtime::options::{BoxOptions, BoxliteOptions, PortSpec, RootfsSpec};
use boxlite::{BoxCommand, BoxliteRuntime, CopyOptions, Execution, LiteBox};
use futures::StreamExt;
use kaos::{
    CurrentKaosToken, SshHostKeyPolicy, SshKaos, SshKaosOptions, reset_current_kaos,
    set_current_kaos,
};
use rusqlite::Connection;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::{Instant, sleep};

pub const GUEST_FIXTURE_DIR: &str = "/root/fixtures";
pub const GUEST_PYTHON: &str = "/root/fixtures/venv/bin/python3";
pub const GUEST_HTTP_SCRIPT: &str = "/root/fixtures/boxlite_mcp_http.py";
pub const GUEST_OAUTH_SCRIPT: &str = "/root/fixtures/boxlite_oauth_mcp.py";
pub const GUEST_SSHD_CONFIG: &str = "/root/fixtures/sshd_config";
pub const REMOTE_WORK_DIR: &str = "/root/kimi-agent-boxlite-e2e-workdir";
pub const GUEST_OAUTH_STATE_FILE: &str = "/tmp/box-oauth-state.json";
pub const HTTP_ENV_VALUE: &str = "fixture-http";
pub const OAUTH_ENV_VALUE: &str = "fixture-oauth";

const PYTHON_MCP_COMMON: &str = include_str!("../fixtures/boxlite_e2e/boxlite_mcp_common.py");
const PYTHON_MCP_STDIO: &str = include_str!("../fixtures/boxlite_e2e/boxlite_mcp_stdio.py");
const PYTHON_MCP_HTTP: &str = include_str!("../fixtures/boxlite_e2e/boxlite_mcp_http.py");
const PYTHON_MCP_OAUTH: &str = include_str!("../fixtures/boxlite_e2e/boxlite_oauth_mcp.py");
const PYTHON_REQUIREMENTS: &str = include_str!("../fixtures/boxlite_e2e/requirements.txt");
const SSHD_CONFIG: &str = include_str!("../fixtures/boxlite_e2e/sshd_config");
const START_SSHD_SCRIPT: &str = include_str!("../fixtures/boxlite_e2e/start_sshd.sh");
const START_HTTP_FIXTURE_SCRIPT: &str =
    include_str!("../fixtures/boxlite_e2e/start_http_fixture.sh");
const START_OAUTH_FIXTURE_SCRIPT: &str =
    include_str!("../fixtures/boxlite_e2e/start_oauth_fixture.sh");
const DEBUG_TMUX_ENV: &str = "KIMI_BOXLITE_DEBUG_TMUX";
const SSHD_LOG_PATH: &str = "/tmp/sshd.log";
const HTTP_LOG_PATH: &str = "/tmp/box-http.log";
const OAUTH_LOG_PATH: &str = "/tmp/box-oauth.log";
const APK_LOG_PATH: &str = "/tmp/apk.log";
const PIP_LOG_PATH: &str = "/tmp/boxlite-pip.log";
const TMUX_SSHD_SESSION: &str = "box-sshd";
const TMUX_HTTP_SESSION: &str = "box-http";
const TMUX_OAUTH_SESSION: &str = "box-oauth";
const HTTP_TRANSPORT_VALUE: &str = "http";
const GUEST_PACKAGE_INSTALL_TIMEOUT: Duration = Duration::from_secs(900);
const GUEST_PYTHON_INSTALL_TIMEOUT: Duration = Duration::from_secs(900);
const ALPINE_MIRROR_HOST: &str = "https://mirrors.aliyun.com/alpine";

pub struct ProcessOutput {
    pub stdout: String,
}

struct BoxCommandOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

enum GuestServiceHandle {
    Direct(Execution),
    Tmux { session: &'static str },
}

pub struct BoxliteServices {
    pub http: bool,
    pub oauth: bool,
}

pub(crate) struct GuestService {
    pub(crate) guest_port: u16,
    pub(crate) host_forward_port: Option<u16>,
    label: &'static str,
    tmux_session: &'static str,
    log_path: &'static str,
    handle: GuestServiceHandle,
}

/// End-to-end BoxLite fixture that boots a real SSH server plus optional real Python MCP
/// services inside an isolated micro VM. Tests opt into only the guest services they need so the
/// fixture surface stays small and per-target support code does not accumulate dead code.
pub struct BoxliteSshFixture {
    runtime: BoxliteRuntime,
    pub(crate) litebox: LiteBox,
    host_home: TempDir,
    cli_config_path: PathBuf,
    private_key: String,
    ssh_host_port: u16,
    debug_tmux: bool,
    pub(crate) http: Option<GuestService>,
    pub(crate) oauth: Option<GuestService>,
    sshd: GuestServiceHandle,
}

pub struct CurrentKaosGuard {
    token: Option<CurrentKaosToken>,
}

impl CurrentKaosGuard {
    pub fn new(kaos: Arc<dyn kaos::Kaos>) -> Self {
        Self {
            token: Some(set_current_kaos(kaos)),
        }
    }
}

impl Drop for CurrentKaosGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            reset_current_kaos(token);
        }
    }
}

impl GuestServiceHandle {
    async fn stop(&mut self, litebox: &LiteBox) -> Result<()> {
        match self {
            Self::Direct(execution) => execution.kill().await.context("kill guest process"),
            Self::Tmux { session } => {
                let _ = exec_box_output(
                    litebox,
                    BoxCommand::new("sh").args(["-lc", &format!("tmux kill-session -t {session}")]),
                )
                .await;
                Ok(())
            }
        }
    }
}

impl BoxliteSshFixture {
    pub async fn provision(services: BoxliteServices) -> Result<Self> {
        let debug_tmux = debug_tmux_enabled();
        let runtime_home = shared_runtime_home()?;
        let host_home = TempDir::new().context("create host home")?;
        let ssh_host_port = reserve_tcp_port()?;
        let guest_http_port = if services.http {
            Some(reserve_unique_port(&[ssh_host_port])?)
        } else {
            None
        };
        let guest_oauth_port = if services.oauth {
            Some(reserve_unique_port(
                &[ssh_host_port]
                    .into_iter()
                    .chain(guest_http_port)
                    .collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        let host_oauth_port = if let Some(guest_oauth_port) = guest_oauth_port {
            Some(reserve_unique_port(&[
                ssh_host_port,
                guest_http_port.unwrap_or(0),
                guest_oauth_port,
            ])?)
        } else {
            None
        };

        let (private_key, public_key) = generate_ssh_keypair(host_home.path())?;
        let guest_root = write_guest_root(host_home.path(), &public_key)?;
        let cli_config_path = write_cli_config(host_home.path(), ssh_host_port, &private_key)?;

        let runtime = BoxliteRuntime::new(BoxliteOptions {
            home_dir: runtime_home,
            image_registries: Vec::new(),
        })
        .context("create BoxLite runtime")?;

        let mut forwarded_ports = vec![PortSpec {
            host_port: Some(ssh_host_port),
            guest_port: 22,
            protocol: Default::default(),
            host_ip: Some("127.0.0.1".to_string()),
        }];
        if let (Some(host_oauth_port), Some(guest_oauth_port)) = (host_oauth_port, guest_oauth_port)
        {
            forwarded_ports.push(PortSpec {
                host_port: Some(host_oauth_port),
                guest_port: guest_oauth_port,
                protocol: Default::default(),
                host_ip: Some("127.0.0.1".to_string()),
            });
        }

        let litebox = runtime
            .create(
                BoxOptions {
                    rootfs: RootfsSpec::Image("alpine:3.20".to_string()),
                    ports: forwarded_ports,
                    ..Default::default()
                },
                Some(format!("kimi-agent-mcp-boxlite-e2e-{ssh_host_port}")),
            )
            .await
            .context("create BoxLite box")?;
        litebox.start().await.context("start BoxLite box")?;

        configure_guest_apk(&litebox).await?;
        install_guest_packages(&litebox, debug_tmux).await?;

        litebox
            .copy_into(
                &guest_root,
                "/root",
                CopyOptions::default().include_parent(false),
            )
            .await
            .context("copy guest fixtures into the box")?;

        install_guest_python_dependencies(&litebox).await?;
        prepare_guest_filesystem(&litebox).await?;

        let sshd = start_guest_sshd(&litebox, debug_tmux).await?;
        let http = if let Some(guest_port) = guest_http_port {
            Some(start_guest_http_fixture(&litebox, guest_port, debug_tmux).await?)
        } else {
            None
        };
        let oauth = if let (Some(guest_port), Some(host_forward_port)) =
            (guest_oauth_port, host_oauth_port)
        {
            Some(
                start_guest_oauth_fixture(&litebox, guest_port, host_forward_port, debug_tmux)
                    .await?,
            )
        } else {
            None
        };

        let fixture = Self {
            runtime,
            litebox,
            host_home,
            cli_config_path,
            private_key,
            ssh_host_port,
            debug_tmux,
            http,
            oauth,
            sshd,
        };

        if let Some(service) = &fixture.http {
            fixture
                .wait_for_guest_service(service.guest_port, service.label)
                .await?;
        }
        if let Some(service) = &fixture.oauth {
            fixture
                .wait_for_guest_service(service.guest_port, service.label)
                .await?;
        }
        fixture.wait_for_ssh().await?;

        Ok(fixture)
    }

    pub fn cli_config_path(&self) -> &Path {
        &self.cli_config_path
    }

    pub fn host_home(&self) -> &Path {
        self.host_home.path()
    }

    pub fn local_state_db_path(&self) -> PathBuf {
        self.host_home().join(".kimi").join("state.db")
    }

    pub fn with_local_state_db<T, F>(&self, op: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let path = self.local_state_db_path();
        let conn = Connection::open(&path)
            .with_context(|| format!("open local SQLite state database {}", path.display()))?;
        op(&conn)
    }

    pub async fn remote_file_exists(&self, path: &str) -> Result<bool> {
        let output = exec_box_output(
            &self.litebox,
            BoxCommand::new("sh").args(["-lc", &format!("test -e {path}")]),
        )
        .await
        .with_context(|| format!("probe remote file existence for {path}"))?;
        Ok(output.exit_code == 0)
    }

    pub async fn connect_ssh_kaos(&self) -> Result<SshKaos> {
        SshKaos::connect(self.ssh_options(Duration::from_secs(10)))
            .await
            .context("connect to BoxLite ssh backend")
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(service) = self.oauth.as_mut() {
            let _ = service.handle.stop(&self.litebox).await;
        }
        if let Some(service) = self.http.as_mut() {
            let _ = service.handle.stop(&self.litebox).await;
        }
        let _ = self.sshd.stop(&self.litebox).await;
        self.runtime
            .shutdown(Some(20))
            .await
            .map_err(|err| anyhow!("shutdown BoxLite runtime: {err}"))?;
        Ok(())
    }

    pub async fn dump_debug_artifacts(&self) {
        self.dump_guest_command_output("guest process list", "ps -o pid,ppid,etime,args")
            .await;

        if self.debug_tmux {
            self.dump_guest_command_output("tmux sessions", "tmux ls")
                .await;
            self.dump_guest_command_output(
                "tmux sshd pane",
                &format!("tmux capture-pane -p -t {TMUX_SSHD_SESSION}:0"),
            )
            .await;
            if let Some(service) = &self.http {
                self.dump_guest_command_output(
                    &format!("tmux {} pane", service.label),
                    &format!("tmux capture-pane -p -t {}:0", service.tmux_session),
                )
                .await;
            }
            if let Some(service) = &self.oauth {
                self.dump_guest_command_output(
                    &format!(
                        "tmux {} pane (host 127.0.0.1:{} -> guest 127.0.0.1:{})",
                        service.label,
                        service.host_forward_port.unwrap_or(service.guest_port),
                        service.guest_port
                    ),
                    &format!("tmux capture-pane -p -t {}:0", service.tmux_session),
                )
                .await;
            }
        }

        self.dump_guest_command_output("sshd log", &format!("tail -n 200 {SSHD_LOG_PATH} || true"))
            .await;
        if let Some(service) = &self.http {
            self.dump_guest_command_output(
                &format!("{} log", service.label),
                &format!("tail -n 200 {} || true", service.log_path),
            )
            .await;
        }
        if let Some(service) = &self.oauth {
            self.dump_guest_command_output(
                &format!(
                    "{} log (host 127.0.0.1:{} -> guest 127.0.0.1:{})",
                    service.label,
                    service.host_forward_port.unwrap_or(service.guest_port),
                    service.guest_port
                ),
                &format!("tail -n 200 {} || true", service.log_path),
            )
            .await;
            self.dump_guest_command_output(
                "oauth state",
                &format!("cat {GUEST_OAUTH_STATE_FILE} || true"),
            )
            .await;
        }
    }

    pub fn ssh_options(&self, connect_timeout: Duration) -> SshKaosOptions {
        SshKaosOptions {
            host: "127.0.0.1".to_string(),
            port: self.ssh_host_port,
            username: Some("root".to_string()),
            password: None,
            key_paths: Vec::new(),
            key_contents: vec![self.private_key.clone()],
            cwd: Some("/root".to_string()),
            known_hosts_path: None,
            host_key_policy: SshHostKeyPolicy::Insecure,
            connect_timeout_seconds: connect_timeout.as_secs().max(1),
        }
    }

    async fn wait_for_guest_service(&self, guest_port: u16, label: &str) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let command = format!(
            "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{guest_port}/health').read().decode(), end='')"
        );
        loop {
            match exec_box_checked(
                &self.litebox,
                // Probe with the same interpreter environment that serves MCP so missing wheels or
                // import errors fail early during fixture bootstrap instead of later during tests.
                BoxCommand::new(GUEST_PYTHON).args(["-c", command.as_str()]),
            )
            .await
            {
                Ok(stdout) if stdout.trim() == "ok" => return Ok(()),
                Ok(_) | Err(_) if Instant::now() < deadline => {
                    sleep(Duration::from_millis(250)).await;
                }
                Ok(stdout) => {
                    bail!(
                        "guest {label} fixture became reachable but returned unexpected payload: {stdout}"
                    );
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("guest {label} fixture did not become ready"));
                }
            }
        }
    }

    async fn wait_for_ssh(&self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match SshKaos::connect(self.ssh_options(Duration::from_secs(2))).await {
                Ok(_) => return Ok(()),
                Err(_) if Instant::now() < deadline => {
                    sleep(Duration::from_millis(250)).await;
                }
                Err(err) => {
                    return Err(err).context("guest sshd did not become ready");
                }
            }
        }
    }

    async fn dump_guest_command_output(&self, label: &str, command: &str) {
        match exec_box_output(&self.litebox, BoxCommand::new("sh").args(["-lc", command])).await {
            Ok(output) => {
                eprintln!("--- {label} (exit {}) ---", output.exit_code);
                if !output.stdout.is_empty() {
                    eprintln!("{}", output.stdout);
                }
                if !output.stderr.is_empty() {
                    eprintln!("{}", output.stderr);
                }
            }
            Err(err) => {
                eprintln!("--- {label} unavailable ---");
                eprintln!("{err:#}");
            }
        }
    }
}

pub async fn exec_box_checked(litebox: &LiteBox, command: BoxCommand) -> Result<String> {
    let output = exec_box_output(litebox, command).await?;
    if output.exit_code != 0 {
        bail!(
            "guest command exited with code {}\nstdout:\n{}\nstderr:\n{}",
            output.exit_code,
            output.stdout,
            output.stderr
        );
    }
    Ok(output.stdout)
}

pub fn run_kimi_agent(
    host_home: &Path,
    config_file: &Path,
    args: &[&str],
) -> Result<ProcessOutput> {
    run_kimi_agent_with_env(host_home, config_file, &[], args)
}

pub fn run_kimi_agent_with_env(
    host_home: &Path,
    config_file: &Path,
    envs: &[(&str, &str)],
    args: &[&str],
) -> Result<ProcessOutput> {
    const CLI_TIMEOUT: Duration = Duration::from_secs(45);

    let mut command = Command::new(env!("CARGO_BIN_EXE_kimi-agent"));
    command
        .env("HOME", host_home)
        .arg("--config-file")
        .arg(config_file)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("spawn kimi-agent with args: {args:?}"))?;

    let started_at = std::time::Instant::now();
    let output = loop {
        if child
            .try_wait()
            .with_context(|| format!("poll kimi-agent with args: {args:?}"))?
            .is_some()
        {
            break child
                .wait_with_output()
                .with_context(|| format!("collect kimi-agent output for args: {args:?}"))?;
        }

        if started_at.elapsed() >= CLI_TIMEOUT {
            let _ = child.kill();
            let output = child.wait_with_output().with_context(|| {
                format!("collect timed out kimi-agent output for args: {args:?}")
            })?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "kimi-agent {:?} timed out after {:?}\nstdout:\n{}\nstderr:\n{}",
                args,
                CLI_TIMEOUT,
                stdout,
                stderr
            );
        }

        std::thread::sleep(Duration::from_millis(100));
    };

    let stdout = String::from_utf8(output.stdout).context("decode kimi-agent stdout")?;
    let stderr = String::from_utf8(output.stderr).context("decode kimi-agent stderr")?;

    if !output.status.success() {
        bail!(
            "kimi-agent {:?} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            args,
            output.status.code(),
            stdout,
            stderr
        );
    }

    Ok(ProcessOutput { stdout })
}

async fn exec_box_output(litebox: &LiteBox, command: BoxCommand) -> Result<BoxCommandOutput> {
    let mut execution = litebox
        .exec(command)
        .await
        .context("exec in BoxLite guest")?;
    let stdout_stream = execution.stdout();
    let stderr_stream = execution.stderr();

    let stdout_task = tokio::spawn(collect_stream(stdout_stream));
    let stderr_task = tokio::spawn(collect_stream(stderr_stream));

    let result = execution.wait().await.context("wait for BoxLite command")?;
    let stdout = stdout_task.await.context("join stdout task")?;
    let stderr = stderr_task.await.context("join stderr task")?;

    Ok(BoxCommandOutput {
        exit_code: result.exit_code,
        stdout,
        stderr,
    })
}

async fn collect_stream<S>(stream: Option<S>) -> String
where
    S: futures::Stream<Item = String> + Send + Unpin + 'static,
{
    let Some(mut stream) = stream else {
        return String::new();
    };

    let mut output = String::new();
    while let Some(chunk) = stream.next().await {
        output.push_str(&chunk);
    }
    output
}

fn shared_runtime_home() -> Result<PathBuf> {
    let mut base = dirs::home_dir().unwrap_or_else(std::env::temp_dir);
    base.push(".boxlite-it-kimi-agent");
    base.push("shared-runtime");
    std::fs::create_dir_all(&base).with_context(|| format!("create {}", base.display()))?;
    Ok(base)
}

fn reserve_unique_port(used_ports: &[u16]) -> Result<u16> {
    loop {
        let port = reserve_tcp_port()?;
        if !used_ports.contains(&port) {
            return Ok(port);
        }
    }
}

fn reserve_tcp_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).context("bind ephemeral port")?;
    let port = listener.local_addr().context("read ephemeral port")?.port();
    drop(listener);
    Ok(port)
}

fn generate_ssh_keypair(host_home: &Path) -> Result<(String, String)> {
    let key_path = host_home.join("ssh-key");
    let status = Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-f"])
        .arg(&key_path)
        .status()
        .context("run ssh-keygen")?;
    if !status.success() {
        bail!("ssh-keygen failed with status {:?}", status.code());
    }

    let private_key = read_to_string(&key_path)?;
    let public_key = read_to_string(&key_path.with_extension("pub"))?;
    Ok((private_key, public_key))
}

fn write_guest_root(host_home: &Path, public_key: &str) -> Result<PathBuf> {
    let guest_root = host_home.join("guest-root");
    let fixtures_dir = guest_root.join("fixtures");
    let ssh_dir = guest_root.join(".ssh");

    std::fs::create_dir_all(&fixtures_dir)
        .with_context(|| format!("create {}", fixtures_dir.display()))?;
    std::fs::create_dir_all(&ssh_dir).with_context(|| format!("create {}", ssh_dir.display()))?;

    write_string(
        fixtures_dir.join("boxlite_mcp_common.py"),
        PYTHON_MCP_COMMON,
    )?;
    write_string(fixtures_dir.join("boxlite_mcp_stdio.py"), PYTHON_MCP_STDIO)?;
    write_string(fixtures_dir.join("boxlite_mcp_http.py"), PYTHON_MCP_HTTP)?;
    write_string(fixtures_dir.join("boxlite_oauth_mcp.py"), PYTHON_MCP_OAUTH)?;
    write_string(fixtures_dir.join("requirements.txt"), PYTHON_REQUIREMENTS)?;
    write_string(fixtures_dir.join("sshd_config"), SSHD_CONFIG)?;
    write_string(fixtures_dir.join("start_sshd.sh"), START_SSHD_SCRIPT)?;
    write_string(
        fixtures_dir.join("start_http_fixture.sh"),
        START_HTTP_FIXTURE_SCRIPT,
    )?;
    write_string(
        fixtures_dir.join("start_oauth_fixture.sh"),
        START_OAUTH_FIXTURE_SCRIPT,
    )?;
    write_string(ssh_dir.join("authorized_keys"), public_key)?;

    Ok(guest_root)
}

fn write_cli_config(host_home: &Path, ssh_host_port: u16, private_key: &str) -> Result<PathBuf> {
    let config_path = host_home.join("kimi-agent-boxlite-config.json");
    let config = json!({
        "kaos": {
            "type": "ssh",
            "host": "127.0.0.1",
            "port": ssh_host_port,
            "username": "root",
            "key_contents": [private_key],
            "cwd": "/root",
            "host_key_policy": "insecure",
            "connect_timeout_seconds": 10
        },
        "mcp": {
            "client": {
                "tool_call_timeout_ms": 10000
            }
        }
    });
    let text = serde_json::to_string_pretty(&config).context("serialize CLI config")?;
    write_string(&config_path, &text)?;
    Ok(config_path)
}

fn write_string(path: impl AsRef<Path>, contents: &str) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))
}

fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

async fn configure_guest_apk(litebox: &LiteBox) -> Result<()> {
    exec_box_checked(
        litebox,
        BoxCommand::new("sh").args([
            "-lc",
            &format!(
                "release=\"$(cut -d. -f1,2 /etc/alpine-release)\" && \
                 printf '%s\\n%s\\n' \
                 '{mirror}/v'$release'/main' \
                 '{mirror}/v'$release'/community' \
                 > /etc/apk/repositories",
                mirror = ALPINE_MIRROR_HOST,
            ),
        ]),
    )
    .await
    .context("configure guest apk mirror")?;
    Ok(())
}

async fn install_guest_packages(litebox: &LiteBox, debug_tmux: bool) -> Result<()> {
    let guest_packages = if debug_tmux {
        "python3 py3-pip py3-virtualenv openssh-server ca-certificates tmux"
    } else {
        "python3 py3-pip py3-virtualenv openssh-server ca-certificates"
    };

    if let Err(err) = exec_box_checked(
        litebox,
        BoxCommand::new("sh")
            .args([
                "-lc",
                &format!("apk add --no-cache {guest_packages} >{APK_LOG_PATH} 2>&1"),
            ])
            .timeout(GUEST_PACKAGE_INSTALL_TIMEOUT),
    )
    .await
    {
        let log_tail = read_guest_log_tail(litebox, APK_LOG_PATH, 120).await;
        return Err(err).context(format!(
            "install guest dependencies\nrecent guest log {APK_LOG_PATH}:\n{log_tail}"
        ));
    }

    Ok(())
}

async fn prepare_guest_filesystem(litebox: &LiteBox) -> Result<()> {
    exec_box_checked(
        litebox,
        BoxCommand::new("sh").args([
            "-lc",
            "mkdir -p /run/sshd /root/.ssh /root/kimi-agent-boxlite-e2e-workdir && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && chmod +x /root/fixtures/*.sh && ssh-keygen -A",
        ]),
    )
    .await
    .context("prepare guest sshd state")?;
    Ok(())
}

async fn install_guest_python_dependencies(litebox: &LiteBox) -> Result<()> {
    const MAX_ATTEMPTS: usize = 3;
    const PIP_INDEX_URL: &str = "https://mirrors.aliyun.com/pypi/simple/";
    let install_command = format!(
        "(rm -rf {fixture_dir}/venv && \
          python3 -m virtualenv {fixture_dir}/venv && \
          {python} -m pip install --disable-pip-version-check --no-cache-dir --retries 5 \
          --index-url {index_url} --trusted-host mirrors.aliyun.com \
          -r {fixture_dir}/requirements.txt) >{log_path} 2>&1",
        fixture_dir = GUEST_FIXTURE_DIR,
        python = GUEST_PYTHON,
        index_url = PIP_INDEX_URL,
        log_path = PIP_LOG_PATH,
    );

    let mut last_error = None;
    for attempt in 1..=MAX_ATTEMPTS {
        match exec_box_checked(
            litebox,
            BoxCommand::new("sh")
                .args(["-lc", install_command.as_str()])
                .timeout(GUEST_PYTHON_INSTALL_TIMEOUT),
        )
        .await
        {
            Ok(_) => return Ok(()),
            Err(err) if attempt < MAX_ATTEMPTS => {
                let log_tail = read_guest_log_tail(litebox, PIP_LOG_PATH, 120).await;
                eprintln!(
                    "guest Python dependency installation attempt {attempt}/{MAX_ATTEMPTS} failed: {err:#}\nrecent guest log {PIP_LOG_PATH}:\n{log_tail}"
                );
                last_error = Some(err);
                sleep(Duration::from_secs(2)).await;
            }
            Err(err) => {
                let log_tail = read_guest_log_tail(litebox, PIP_LOG_PATH, 120).await;
                last_error =
                    Some(err.context(format!("recent guest log {PIP_LOG_PATH}:\n{log_tail}")));
                break;
            }
        }
    }

    Err(last_error.unwrap()).context("install guest Python MCP dependency")
}

async fn start_guest_sshd(litebox: &LiteBox, debug_tmux: bool) -> Result<GuestServiceHandle> {
    if debug_tmux {
        start_tmux_session(litebox, TMUX_SSHD_SESSION, "/root/fixtures/start_sshd.sh").await?;
        Ok(GuestServiceHandle::Tmux {
            session: TMUX_SSHD_SESSION,
        })
    } else {
        let execution = litebox
            .exec(BoxCommand::new("/usr/sbin/sshd").args([
                "-D",
                "-f",
                GUEST_SSHD_CONFIG,
                "-E",
                SSHD_LOG_PATH,
            ]))
            .await
            .context("start guest sshd")?;
        Ok(GuestServiceHandle::Direct(execution))
    }
}

async fn start_guest_http_fixture(
    litebox: &LiteBox,
    guest_port: u16,
    debug_tmux: bool,
) -> Result<GuestService> {
    let handle = if debug_tmux {
        let command = format!(
            "env MCP_HTTP_PORT={guest_port} BOX_MCP_ENV={HTTP_ENV_VALUE} FIXTURE_TRANSPORT={HTTP_TRANSPORT_VALUE} /root/fixtures/start_http_fixture.sh"
        );
        start_tmux_session(litebox, TMUX_HTTP_SESSION, &command).await?;
        GuestServiceHandle::Tmux {
            session: TMUX_HTTP_SESSION,
        }
    } else {
        GuestServiceHandle::Direct(
            litebox
                .exec(
                    BoxCommand::new(GUEST_PYTHON)
                        .arg(GUEST_HTTP_SCRIPT)
                        .env("MCP_HTTP_PORT", guest_port.to_string())
                        .env("BOX_MCP_ENV", HTTP_ENV_VALUE)
                        .env("FIXTURE_TRANSPORT", HTTP_TRANSPORT_VALUE)
                        .working_dir(GUEST_FIXTURE_DIR),
                )
                .await
                .context("start guest HTTP MCP fixture")?,
        )
    };

    Ok(GuestService {
        guest_port,
        host_forward_port: None,
        label: "http",
        tmux_session: TMUX_HTTP_SESSION,
        log_path: HTTP_LOG_PATH,
        handle,
    })
}

async fn start_guest_oauth_fixture(
    litebox: &LiteBox,
    guest_port: u16,
    host_forward_port: u16,
    debug_tmux: bool,
) -> Result<GuestService> {
    let handle = if debug_tmux {
        let command = format!(
            "env MCP_OAUTH_PORT={guest_port} BOX_MCP_ENV={OAUTH_ENV_VALUE} FIXTURE_TRANSPORT=oauth-http OAUTH_STATE_PATH={GUEST_OAUTH_STATE_FILE} /root/fixtures/start_oauth_fixture.sh 2>&1 | tee {OAUTH_LOG_PATH}"
        );
        start_tmux_session(litebox, TMUX_OAUTH_SESSION, &command).await?;
        GuestServiceHandle::Tmux {
            session: TMUX_OAUTH_SESSION,
        }
    } else {
        GuestServiceHandle::Direct(
            litebox
                .exec(
                    BoxCommand::new(GUEST_PYTHON)
                        .arg(GUEST_OAUTH_SCRIPT)
                        .env("MCP_OAUTH_PORT", guest_port.to_string())
                        .env("BOX_MCP_ENV", OAUTH_ENV_VALUE)
                        .env("FIXTURE_TRANSPORT", "oauth-http")
                        .env("OAUTH_STATE_PATH", GUEST_OAUTH_STATE_FILE)
                        .working_dir(GUEST_FIXTURE_DIR),
                )
                .await
                .context("start guest OAuth MCP fixture")?,
        )
    };

    Ok(GuestService {
        guest_port,
        host_forward_port: Some(host_forward_port),
        label: "oauth",
        tmux_session: TMUX_OAUTH_SESSION,
        log_path: OAUTH_LOG_PATH,
        handle,
    })
}

async fn start_tmux_session(litebox: &LiteBox, session: &'static str, command: &str) -> Result<()> {
    let _ = exec_box_output(
        litebox,
        BoxCommand::new("tmux").args(["kill-session", "-t", session]),
    )
    .await;

    exec_box_checked(
        litebox,
        BoxCommand::new("tmux").args(["new-session", "-d", "-s", session, command]),
    )
    .await
    .with_context(|| format!("start tmux session {session}"))?;
    Ok(())
}

async fn read_guest_log_tail(litebox: &LiteBox, path: &str, lines: usize) -> String {
    let command = format!("tail -n {lines} {path} 2>/dev/null || true");
    exec_box_output(
        litebox,
        BoxCommand::new("sh").args(["-lc", command.as_str()]),
    )
    .await
    .map(|output| output.stdout)
    .unwrap_or_default()
}

fn debug_tmux_enabled() -> bool {
    matches!(
        std::env::var(DEBUG_TMUX_ENV).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}
