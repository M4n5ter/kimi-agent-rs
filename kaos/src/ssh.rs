use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use russh::keys::known_hosts::learn_known_hosts_path;
use russh::keys::{
    PrivateKeyWithHashAlg, check_known_hosts_path, decode_secret_key, load_secret_key,
};
use russh::{ChannelMsg, Sig, client};
use russh_sftp::client::SftpSession;
use russh_sftp::client::error::Error as SftpError;
use russh_sftp::protocol::{OpenFlags, StatusCode};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex as AsyncMutex, mpsc, watch};
use typed_path::Utf8TypedPathBuf;

use crate::{
    AsyncReadWrite, AsyncReadable, AsyncWritable, ExecOptions, Kaos, KaosFileError,
    KaosFileErrorKind, KaosPath, KaosPathStyle, KaosPlatform, KaosProcess, LineStream,
    ProcessOutputOverflow, StatResult, StrOrKaosPath, line_stream::line_stream_from_async_read,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SshHostKeyPolicy {
    Strict,
    #[default]
    AcceptNew,
    Insecure,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SshKaosOptions {
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_contents: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_hosts_path: Option<String>,
    #[serde(default)]
    pub host_key_policy: SshHostKeyPolicy,
    #[serde(default = "default_connect_timeout_seconds")]
    pub connect_timeout_seconds: u64,
}

const fn default_port() -> u16 {
    22
}

const fn default_connect_timeout_seconds() -> u64 {
    15
}

const MAX_AGENT_IDENTITIES_TO_TRY: usize = 8;
const PROCESS_OUTPUT_QUEUE_CAPACITY: usize = 64;
const DEFAULT_SSH_EXIT_CODE: i32 = 1;

#[derive(Clone, Copy)]
enum ProcessStreamKind {
    Stdout,
    Stderr,
}

impl SshKaosOptions {
    pub fn logical_storage_name(&self) -> String {
        let username = self.username.clone().unwrap_or_else(default_ssh_username);
        build_storage_name(&self.host, self.port, &username)
    }

    fn known_hosts_path(&self) -> PathBuf {
        let configured = self
            .known_hosts_path
            .clone()
            .unwrap_or_else(|| "~/.kimi/known_hosts".to_string());
        expand_home(&configured)
    }
}

#[derive(Clone)]
pub struct SshKaos {
    state: Arc<SshState>,
}

struct SshState {
    handle: AsyncMutex<client::Handle<SshClientHandler>>,
    sftp: SftpSession,
    home: KaosPath,
    cwd: Mutex<KaosPath>,
    env_cache: AsyncMutex<HashMap<String, Option<String>>>,
    platform: KaosPlatform,
    storage_name: String,
}

struct SshClientHandler {
    host: String,
    port: u16,
    host_key_policy: SshHostKeyPolicy,
    known_hosts_path: PathBuf,
}

impl client::Handler for SshClientHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh::keys::PublicKey,
    ) -> Result<bool, Self::Error> {
        match self.host_key_policy {
            SshHostKeyPolicy::Insecure => Ok(true),
            SshHostKeyPolicy::Strict => {
                let matched = check_known_hosts_path(
                    &self.host,
                    self.port,
                    server_public_key,
                    &self.known_hosts_path,
                )?;
                if matched {
                    Ok(true)
                } else {
                    bail!(
                        "Host key for {}:{} is not trusted (policy: strict).",
                        self.host,
                        self.port
                    );
                }
            }
            SshHostKeyPolicy::AcceptNew => {
                match check_known_hosts_path(
                    &self.host,
                    self.port,
                    server_public_key,
                    &self.known_hosts_path,
                ) {
                    Ok(true) => Ok(true),
                    Ok(false) => {
                        learn_known_hosts_path(
                            &self.host,
                            self.port,
                            server_public_key,
                            &self.known_hosts_path,
                        )?;
                        Ok(true)
                    }
                    Err(russh::keys::Error::KeyChanged { line }) => bail!(
                        "Host key changed for {}:{} (known_hosts line {}).",
                        self.host,
                        self.port,
                        line
                    ),
                    Err(err) => Err(err.into()),
                }
            }
        }
    }
}

impl SshKaos {
    pub async fn connect(options: SshKaosOptions) -> Result<Self> {
        if options.host.trim().is_empty() {
            bail!("SSH host cannot be empty");
        }

        let host = options.host.clone();
        let port = options.port;
        let username = options
            .username
            .clone()
            .unwrap_or_else(default_ssh_username);
        let known_hosts_path = options.known_hosts_path();
        if let Some(parent) = known_hosts_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connect_timeout = Duration::from_secs(options.connect_timeout_seconds);
        let deadline = tokio::time::Instant::now() + connect_timeout;

        let config = Arc::new(client::Config {
            // Keep session idle behavior independent from connection timeout.
            inactivity_timeout: None,
            ..Default::default()
        });
        let handler = SshClientHandler {
            host: host.clone(),
            port,
            host_key_policy: options.host_key_policy,
            known_hosts_path,
        };

        let (mut handle, sftp, home, cwd) = tokio::time::timeout_at(deadline, async {
            let mut handle = client::connect(config, (host.as_str(), port), handler).await?;
            authenticate(&mut handle, &username, &options).await?;

            let sftp_channel = handle.channel_open_session().await?;
            sftp_channel.request_subsystem(true, "sftp").await?;
            let sftp = SftpSession::new(sftp_channel.into_stream()).await?;

            let home = KaosPath::from_style(
                KaosPathStyle::Posix,
                sftp.canonicalize(".").await.map_err(|err| {
                    sftp_error(
                        &KaosPath::from_style(KaosPathStyle::Posix, "."),
                        "canonicalize",
                        err,
                    )
                })?,
            );
            let cwd = if let Some(cwd) = options.cwd.as_deref() {
                let resolved = resolve_absolute_posix(&home, cwd);
                let requested_cwd = KaosPath::from_style(KaosPathStyle::Posix, &resolved);
                let canonical = sftp
                    .canonicalize(&resolved)
                    .await
                    .map_err(|err| sftp_error(&requested_cwd, "canonicalize", err))?;
                let attrs = sftp.metadata(canonical.as_str()).await.map_err(|err| {
                    sftp_error(
                        &KaosPath::from_style(KaosPathStyle::Posix, canonical.as_str()),
                        "stat",
                        err,
                    )
                })?;
                if !attrs.is_dir() {
                    bail!("Configured SSH cwd is not a directory: {canonical}");
                }
                KaosPath::from_style(KaosPathStyle::Posix, canonical)
            } else {
                home.clone()
            };
            Ok::<_, anyhow::Error>((handle, sftp, home, cwd))
        })
        .await
        .map_err(|_| {
            anyhow!(
                "Timed out establishing SSH session to {}:{} after {}s",
                host,
                port,
                options.connect_timeout_seconds
            )
        })??;

        let platform = detect_remote_platform_with_deadline(&mut handle, deadline).await?;
        let storage_name = build_storage_name(&host, port, &username);

        Ok(Self {
            state: Arc::new(SshState {
                handle: AsyncMutex::new(handle),
                sftp,
                home,
                cwd: Mutex::new(cwd),
                env_cache: AsyncMutex::new(HashMap::new()),
                platform,
                storage_name,
            }),
        })
    }

    fn normalize_posix(path: &str) -> String {
        Utf8TypedPathBuf::from_unix(path).normalize().to_string()
    }

    fn resolve_path(&self, path: &KaosPath) -> KaosPath {
        let normalized = KaosPath::from_style(KaosPathStyle::Posix, path.to_string_lossy());
        if normalized.is_absolute() {
            return KaosPath::from_style(
                KaosPathStyle::Posix,
                Self::normalize_posix(normalized.as_str()),
            );
        }

        let cwd = self
            .state
            .cwd
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        cwd.joinpath(normalized.as_str()).canonical()
    }
}

#[async_trait::async_trait]
impl Kaos for SshKaos {
    fn name(&self) -> &str {
        "ssh"
    }

    fn storage_name(&self) -> String {
        self.state.storage_name.clone()
    }

    fn platform(&self) -> KaosPlatform {
        self.state.platform.clone()
    }

    fn path_style(&self) -> KaosPathStyle {
        KaosPathStyle::Posix
    }

    fn normpath(&self, path: &StrOrKaosPath<'_>) -> KaosPath {
        let raw = match path {
            StrOrKaosPath::Str(s) => *s,
            StrOrKaosPath::KaosPath(p) => p.as_str(),
        };
        KaosPath::from_style(KaosPathStyle::Posix, Self::normalize_posix(raw))
    }

    fn home(&self) -> KaosPath {
        self.state.home.clone()
    }

    fn cwd(&self) -> KaosPath {
        self.state
            .cwd
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    async fn chdir(&self, path: &KaosPath) -> Result<()> {
        let resolved = self.resolve_path(path);
        let canonical = self
            .state
            .sftp
            .canonicalize(resolved.as_str())
            .await
            .map_err(|err| sftp_error(&resolved, "canonicalize", err))?;
        let attrs = self
            .state
            .sftp
            .metadata(canonical.as_str())
            .await
            .map_err(|err| {
                sftp_error(
                    &KaosPath::from_style(KaosPathStyle::Posix, canonical.as_str()),
                    "stat",
                    err,
                )
            })?;
        if !attrs.is_dir() {
            bail!("Not a directory: {}", canonical);
        }
        *self
            .state
            .cwd
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            KaosPath::from_style(KaosPathStyle::Posix, canonical);
        Ok(())
    }

    async fn stat(&self, path: &KaosPath, follow_symlinks: bool) -> Result<StatResult> {
        let resolved = self.resolve_path(path);
        let attrs = if follow_symlinks {
            self.state
                .sftp
                .metadata(resolved.as_str())
                .await
                .map_err(|err| sftp_error(&resolved, "stat", err))?
        } else {
            self.state
                .sftp
                .symlink_metadata(resolved.as_str())
                .await
                .map_err(|err| sftp_error(&resolved, "stat", err))?
        };

        let st_mode = attrs.permissions.unwrap_or(0);
        Ok(StatResult {
            st_mode,
            st_ino: 0,
            st_dev: 0,
            st_nlink: 0,
            st_uid: attrs.uid.unwrap_or(0),
            st_gid: attrs.gid.unwrap_or(0),
            st_size: attrs.size.unwrap_or(0),
            st_atime: attrs.atime.unwrap_or(0) as f64,
            st_mtime: attrs.mtime.unwrap_or(0) as f64,
            st_ctime: attrs.mtime.unwrap_or(0) as f64,
        })
    }

    async fn iterdir(&self, path: &KaosPath) -> Result<Vec<KaosPath>> {
        let resolved = self.resolve_path(path);
        let mut entries = Vec::new();
        let dir = self
            .state
            .sftp
            .read_dir(resolved.as_str())
            .await
            .map_err(|err| sftp_error(&resolved, "read directory", err))?;
        for entry in dir {
            let full = resolved.joinpath(&entry.file_name());
            entries.push(full);
        }
        Ok(entries)
    }

    async fn glob(
        &self,
        path: &KaosPath,
        pattern: &str,
        case_sensitive: bool,
    ) -> Result<Vec<KaosPath>> {
        if !case_sensitive {
            bail!("Case insensitive glob is not supported in current environment");
        }
        let normalized_pattern = normalize_glob_pattern(pattern);
        let resolved = self.resolve_path(path);
        let matcher = glob::Pattern::new(&normalized_pattern).map_err(|err| anyhow!(err))?;
        let options = glob::MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: true,
        };
        let traversal_plan = GlobTraversalPlan::from_pattern(&normalized_pattern)?;

        let mut stack = vec![(resolved.clone(), 0usize)];
        let mut matched = Vec::new();

        while let Some((dir, dir_depth)) = stack.pop() {
            let entries = self
                .state
                .sftp
                .read_dir(dir.as_str())
                .await
                .map_err(|err| sftp_error(&dir, "read directory", err))?;

            for entry in entries {
                let name = entry.file_name();
                // russh-sftp currently filters "." and ".." in ReadDir, but keep a guard
                // here to remain correct if dependency behavior changes.
                if name == "." || name == ".." {
                    continue;
                }
                let full = dir.joinpath(&name);
                let relative = full
                    .relative_to(&resolved)
                    .unwrap_or_else(|_| KaosPath::from_style(KaosPathStyle::Posix, &name));
                let rel = relative.to_string_lossy();
                if matcher.matches_with(&rel, options) {
                    matched.push(full.clone());
                }
                let next_depth = dir_depth + 1;
                if entry.file_type().is_dir()
                    && traversal_plan.should_descend(&rel, next_depth, &options)
                {
                    stack.push((full, next_depth));
                }
            }
        }

        Ok(matched)
    }

    async fn read_bytes(&self, path: &KaosPath, limit: Option<usize>) -> Result<Vec<u8>> {
        let resolved = self.resolve_path(path);
        let mut data = self
            .state
            .sftp
            .read(resolved.as_str())
            .await
            .map_err(|err| sftp_error(&resolved, "read bytes", err))?;
        if let Some(n) = limit {
            data.truncate(n);
        }
        Ok(data)
    }

    async fn read_text(&self, path: &KaosPath) -> Result<String> {
        let bytes = self.read_bytes(path, None).await?;
        String::from_utf8(bytes).map_err(|err| anyhow!("File is not valid UTF-8: {err}"))
    }

    async fn read_lines(&self, path: &KaosPath) -> Result<Vec<String>> {
        let text = self.read_text(path).await?;
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        Ok(normalized
            .split_inclusive('\n')
            .map(str::to_string)
            .collect())
    }

    async fn read_lines_stream(&self, path: &KaosPath) -> Result<LineStream> {
        let resolved = self.resolve_path(path);
        let file = self
            .state
            .sftp
            .open(resolved.as_str())
            .await
            .map_err(|err| sftp_error(&resolved, "open text stream", err))?;
        Ok(line_stream_from_async_read(file))
    }

    async fn write_bytes(&self, path: &KaosPath, data: &[u8]) -> Result<usize> {
        let resolved = self.resolve_path(path);
        let mut file = self
            .state
            .sftp
            .open_with_flags(
                resolved.as_str(),
                OpenFlags::CREATE | OpenFlags::TRUNCATE | OpenFlags::WRITE,
            )
            .await
            .map_err(|err| sftp_error(&resolved, "write bytes", err))?;
        file.write_all(data).await?;
        file.shutdown().await?;
        Ok(data.len())
    }

    async fn write_text(&self, path: &KaosPath, data: &str, append: bool) -> Result<usize> {
        let resolved = self.resolve_path(path);
        let flags = if append {
            OpenFlags::CREATE | OpenFlags::WRITE | OpenFlags::APPEND
        } else {
            OpenFlags::CREATE | OpenFlags::WRITE | OpenFlags::TRUNCATE
        };
        let mut file = self
            .state
            .sftp
            .open_with_flags(resolved.as_str(), flags)
            .await
            .map_err(|err| sftp_error(&resolved, "write text", err))?;
        file.write_all(data.as_bytes()).await?;
        file.shutdown().await?;
        Ok(data.len())
    }

    async fn chmod(&self, path: &KaosPath, mode: u32) -> Result<()> {
        let resolved = self.resolve_path(path);
        let mut attrs = self
            .state
            .sftp
            .metadata(resolved.as_str())
            .await
            .map_err(|err| sftp_error(&resolved, "stat", err))?;
        let file_type_bits = attrs.permissions.unwrap_or(0) & 0o170000;
        attrs.permissions = Some(file_type_bits | (mode & 0o7777));
        self.state
            .sftp
            .set_metadata(resolved.as_str(), attrs)
            .await
            .map_err(|err| sftp_error(&resolved, "chmod", err))?;
        Ok(())
    }

    async fn mkdir(&self, path: &KaosPath, parents: bool, exist_ok: bool) -> Result<()> {
        let resolved = self.resolve_path(path);
        if !parents {
            return mkdir_once(&self.state.sftp, resolved.as_str(), exist_ok).await;
        }

        let normalized = resolved.canonical();
        let mut current = KaosPath::from_style(KaosPathStyle::Posix, "/");
        let normalized_str = normalized.to_string_lossy();
        let components = normalized_str
            .split('/')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();

        for (idx, component) in components.iter().enumerate() {
            current = current.joinpath(component);
            let allow_existing = if idx + 1 == components.len() {
                exist_ok
            } else {
                true
            };
            mkdir_once(&self.state.sftp, current.as_str(), allow_existing).await?;
        }
        Ok(())
    }

    async fn env_var(&self, key: &str) -> Result<Option<String>> {
        validate_env_var_key(key)?;

        if let Some(cached) = self.state.env_cache.lock().await.get(key).cloned() {
            return Ok(cached);
        }

        let command = format!(
            "if [ \"${{{key}+x}}\" = x ]; then printf '%s' \"${{{key}}}\"; else exit {SSH_ENV_VAR_UNSET_EXIT_CODE}; fi"
        );
        let mut handle = self.state.handle.lock().await;
        let (exit_code, stdout, stderr) = exec_capture_raw(&mut handle, &command).await?;
        let value = map_env_var_lookup_result(key, exit_code, stdout, stderr)?;

        self.state
            .env_cache
            .lock()
            .await
            .insert(key.to_string(), value.clone());
        Ok(value)
    }

    async fn exec(&self, args: &[String], options: ExecOptions) -> Result<Box<dyn KaosProcess>> {
        if args.is_empty() {
            bail!("missing command");
        }
        let ExecOptions { cwd, env_overrides } = options;

        let quoted = args
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let cwd = cwd
            .map(|path| self.resolve_path(&path))
            .unwrap_or_else(|| self.cwd());
        let command = format!("cd {} && {}", shell_quote(cwd.as_str()), quoted);

        let handle = self.state.handle.lock().await;
        let channel = handle.channel_open_session().await?;
        let (mut read_half, write_half) = channel.split();
        // Keep SSH env injection aligned with the Python AsyncSSH backend by using
        // protocol-level "env" requests instead of shell-prefix tricks.
        //
        // Important caveat: common OpenSSH servers reject arbitrary environment
        // variable names unless they are explicitly whitelisted via AcceptEnv in
        // sshd_config. Callers should treat env_overrides on SSH as backend-
        // dependent rather than universally portable.
        for (key, value) in &env_overrides {
            validate_env_var_key(key)?;
            write_half.set_env(true, key, value).await?;
            await_channel_request_reply(
                &mut read_half,
                &format!("set environment variable `{key}`"),
            )
            .await?;
        }
        write_half.exec(true, command).await?;
        await_channel_request_reply(&mut read_half, "execute remote command").await?;
        let write_half = Arc::new(AsyncMutex::new(Some(write_half)));
        let stdin_writer = ChannelStdin {
            write_half: Arc::clone(&write_half),
        };

        let (stdout_tx, stdout_rx) = mpsc::channel::<Vec<u8>>(PROCESS_OUTPUT_QUEUE_CAPACITY);
        let (stderr_tx, stderr_rx) = mpsc::channel::<Vec<u8>>(PROCESS_OUTPUT_QUEUE_CAPACITY);
        let (exit_code_tx, exit_code_rx) = watch::channel(None);
        let overflow_state = Arc::new(ProcessOutputOverflowState::default());
        let overflow_state_bg = Arc::clone(&overflow_state);

        tokio::spawn(async move {
            let mut exit_code: Option<i32> = None;
            while let Some(msg) = read_half.wait().await {
                match msg {
                    ChannelMsg::Data { data } => {
                        if !forward_process_output_chunk(
                            &stdout_tx,
                            data.as_ref(),
                            &overflow_state_bg,
                            ProcessStreamKind::Stdout,
                        ) {
                            break;
                        }
                    }
                    ChannelMsg::ExtendedData { data, ext: 1 } => {
                        if !forward_process_output_chunk(
                            &stderr_tx,
                            data.as_ref(),
                            &overflow_state_bg,
                            ProcessStreamKind::Stderr,
                        ) {
                            break;
                        }
                    }
                    ChannelMsg::ExtendedData { data, .. } => {
                        if !forward_process_output_chunk(
                            &stdout_tx,
                            data.as_ref(),
                            &overflow_state_bg,
                            ProcessStreamKind::Stdout,
                        ) {
                            break;
                        }
                    }
                    ChannelMsg::ExitStatus { exit_status } => {
                        exit_code = Some(exit_status as i32);
                    }
                    ChannelMsg::Close => break,
                    ChannelMsg::Eof => {}
                    _ => {}
                }
            }
            let _ = exit_code_tx.send(Some(exit_code.unwrap_or(DEFAULT_SSH_EXIT_CODE)));
        });

        Ok(Box::new(SshProcess {
            stdin: stdin_writer,
            stdout: Some(QueueReader::new(stdout_rx)),
            stderr: Some(QueueReader::new(stderr_rx)),
            null_stdout: QueueReader::empty(),
            null_stderr: QueueReader::empty(),
            exit_code_rx,
            overflow_state,
            write_half,
        }))
    }

    async fn connect_tcp(&self, host: &str, port: u16) -> Result<Box<dyn AsyncReadWrite>> {
        let handle = self.state.handle.lock().await;
        let channel = handle
            .channel_open_direct_tcpip(host, u32::from(port), "127.0.0.1", 0)
            .await?;
        Ok(Box::new(channel.into_stream()))
    }
}

#[derive(Default)]
struct ProcessOutputOverflowState {
    stdout_dropped_chunks: AtomicU64,
    stdout_dropped_bytes: AtomicU64,
    stderr_dropped_chunks: AtomicU64,
    stderr_dropped_bytes: AtomicU64,
}

impl ProcessOutputOverflowState {
    fn record_drop(&self, stream: ProcessStreamKind, bytes: usize) {
        let bytes = bytes as u64;
        match stream {
            ProcessStreamKind::Stdout => {
                self.stdout_dropped_chunks.fetch_add(1, Ordering::Relaxed);
                self.stdout_dropped_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            ProcessStreamKind::Stderr => {
                self.stderr_dropped_chunks.fetch_add(1, Ordering::Relaxed);
                self.stderr_dropped_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    fn snapshot(&self) -> ProcessOutputOverflow {
        ProcessOutputOverflow {
            stdout_dropped_chunks: self.stdout_dropped_chunks.load(Ordering::Relaxed),
            stdout_dropped_bytes: self.stdout_dropped_bytes.load(Ordering::Relaxed),
            stderr_dropped_chunks: self.stderr_dropped_chunks.load(Ordering::Relaxed),
            stderr_dropped_bytes: self.stderr_dropped_bytes.load(Ordering::Relaxed),
        }
    }
}

struct SshProcess {
    stdin: ChannelStdin,
    stdout: Option<QueueReader>,
    stderr: Option<QueueReader>,
    null_stdout: QueueReader,
    null_stderr: QueueReader,
    exit_code_rx: watch::Receiver<Option<i32>>,
    overflow_state: Arc<ProcessOutputOverflowState>,
    write_half: Arc<AsyncMutex<Option<russh::ChannelWriteHalf<client::Msg>>>>,
}

fn forward_process_output_chunk(
    tx: &mpsc::Sender<Vec<u8>>,
    data: &[u8],
    overflow_state: &ProcessOutputOverflowState,
    stream: ProcessStreamKind,
) -> bool {
    match tx.try_send(data.to_vec()) {
        Ok(()) => true,
        Err(TrySendError::Closed(_)) => false,
        Err(TrySendError::Full(chunk)) => {
            overflow_state.record_drop(stream, chunk.len());
            true
        }
    }
}

#[async_trait::async_trait]
impl KaosProcess for SshProcess {
    fn pid(&self) -> u32 {
        0
    }

    fn returncode(&mut self) -> Option<i32> {
        *self.exit_code_rx.borrow()
    }

    async fn wait(&mut self) -> Result<i32> {
        loop {
            if let Some(code) = self.returncode() {
                return Ok(code);
            }
            if self.exit_code_rx.changed().await.is_err() {
                return Ok(self.returncode().unwrap_or(DEFAULT_SSH_EXIT_CODE));
            }
        }
    }

    async fn kill(&mut self) -> Result<()> {
        let mut lock = self.write_half.lock().await;
        if let Some(write_half) = lock.as_ref() {
            let _ = write_half.signal(Sig::TERM).await;
            let _ = write_half.close().await;
        }
        *lock = None;
        Ok(())
    }

    fn stdin(&mut self) -> &mut dyn AsyncWritable {
        &mut self.stdin
    }

    fn stdout(&mut self) -> &mut dyn AsyncReadable {
        self.stdout
            .as_mut()
            .map(|stream| stream as &mut dyn AsyncReadable)
            .unwrap_or(&mut self.null_stdout)
    }

    fn stderr(&mut self) -> &mut dyn AsyncReadable {
        self.stderr
            .as_mut()
            .map(|stream| stream as &mut dyn AsyncReadable)
            .unwrap_or(&mut self.null_stderr)
    }

    fn take_stdout(&mut self) -> Option<Box<dyn AsyncReadable>> {
        self.stdout
            .take()
            .map(|stream| Box::new(stream) as Box<dyn AsyncReadable>)
    }

    fn take_stderr(&mut self) -> Option<Box<dyn AsyncReadable>> {
        self.stderr
            .take()
            .map(|stream| Box::new(stream) as Box<dyn AsyncReadable>)
    }

    fn output_overflow_summary(&self) -> Option<ProcessOutputOverflow> {
        let summary = self.overflow_state.snapshot();
        summary.has_drops().then_some(summary)
    }
}

struct ChannelStdin {
    write_half: Arc<AsyncMutex<Option<russh::ChannelWriteHalf<client::Msg>>>>,
}

#[async_trait::async_trait]
impl AsyncWritable for ChannelStdin {
    async fn write(&mut self, data: &[u8]) -> Result<()> {
        let lock = self.write_half.lock().await;
        let Some(write_half) = lock.as_ref() else {
            bail!("stdin is closed");
        };
        let mut writer = write_half.make_writer();
        writer.write_all(data).await?;
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        let lock = self.write_half.lock().await;
        if let Some(write_half) = lock.as_ref() {
            let mut writer = write_half.make_writer();
            writer.flush().await?;
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        let mut lock = self.write_half.lock().await;
        if let Some(write_half) = lock.as_ref() {
            let _ = write_half.eof().await;
            let _ = write_half.close().await;
        }
        *lock = None;
        Ok(())
    }
}

struct QueueReader {
    receiver: Option<mpsc::Receiver<Vec<u8>>>,
    buffer: VecDeque<u8>,
    eof: bool,
}

impl QueueReader {
    fn new(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            receiver: Some(receiver),
            buffer: VecDeque::new(),
            eof: false,
        }
    }

    fn empty() -> Self {
        Self {
            receiver: None,
            buffer: VecDeque::new(),
            eof: true,
        }
    }

    async fn fill_buffer(&mut self) {
        while !self.eof && self.buffer.is_empty() {
            self.recv_next_chunk().await;
        }
    }

    async fn recv_next_chunk(&mut self) {
        if self.eof {
            return;
        }
        let Some(receiver) = self.receiver.as_mut() else {
            self.eof = true;
            return;
        };
        match receiver.recv().await {
            Some(bytes) if !bytes.is_empty() => self.buffer.extend(bytes),
            Some(_) => {}
            None => {
                self.eof = true;
                self.receiver = None;
            }
        }
    }
}

struct GlobTraversalPlan {
    max_dir_depth: Option<usize>,
    prefix_segment_patterns: Vec<glob::Pattern>,
}

impl GlobTraversalPlan {
    fn from_pattern(pattern: &str) -> Result<Self> {
        let segments: Vec<&str> = pattern.split('/').collect();
        let first_globstar = segments.iter().position(|segment| *segment == "**");
        let prefix_len = first_globstar.unwrap_or(segments.len());
        let prefix_segment_patterns = segments
            .iter()
            .take(prefix_len)
            .map(|segment| glob::Pattern::new(segment).map_err(|err| anyhow!(err)))
            .collect::<Result<Vec<_>>>()?;
        let max_dir_depth = if first_globstar.is_some() {
            None
        } else {
            Some(segments.len().saturating_sub(1))
        };

        Ok(Self {
            max_dir_depth,
            prefix_segment_patterns,
        })
    }

    fn should_descend(&self, rel_path: &str, depth: usize, options: &glob::MatchOptions) -> bool {
        if let Some(max_dir_depth) = self.max_dir_depth
            && depth > max_dir_depth
        {
            return false;
        }

        let prefix_check_len = depth.min(self.prefix_segment_patterns.len());
        for (idx, segment) in rel_path.split('/').take(prefix_check_len).enumerate() {
            if !self.prefix_segment_patterns[idx].matches_with(segment, *options) {
                return false;
            }
        }
        true
    }
}

#[async_trait::async_trait]
impl AsyncReadable for QueueReader {
    async fn read(&mut self, n: usize) -> Result<Vec<u8>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        self.fill_buffer().await;
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(n.min(self.buffer.len()));
        for _ in 0..n {
            if let Some(byte) = self.buffer.pop_front() {
                out.push(byte);
            } else {
                break;
            }
        }
        Ok(out)
    }

    async fn readline(&mut self) -> Result<Vec<u8>> {
        loop {
            if let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
                let mut out = Vec::with_capacity(pos + 1);
                for _ in 0..=pos {
                    if let Some(byte) = self.buffer.pop_front() {
                        out.push(byte);
                    }
                }
                return Ok(out);
            }

            self.recv_next_chunk().await;
            if self.eof {
                let mut out = Vec::with_capacity(self.buffer.len());
                while let Some(byte) = self.buffer.pop_front() {
                    out.push(byte);
                }
                return Ok(out);
            }
        }
    }

    fn is_eof(&self) -> bool {
        self.eof && self.buffer.is_empty()
    }
}

async fn authenticate(
    handle: &mut client::Handle<SshClientHandler>,
    username: &str,
    options: &SshKaosOptions,
) -> Result<()> {
    let preferred_hash = handle.best_supported_rsa_hash().await?.flatten();

    for key in &options.key_contents {
        let key = decode_secret_key(key, None)?;
        let key = PrivateKeyWithHashAlg::new(Arc::new(key), preferred_hash);
        let result = handle.authenticate_publickey(username, key).await?;
        if result.success() {
            return Ok(());
        }
    }

    let mut explicit_key_paths = HashSet::new();
    for path in &options.key_paths {
        let path = expand_home(path);
        explicit_key_paths.insert(path.clone());
        let key = load_secret_key(&path, None)?;
        let key = PrivateKeyWithHashAlg::new(Arc::new(key), preferred_hash);
        let result = handle.authenticate_publickey(username, key).await?;
        if result.success() {
            return Ok(());
        }
    }

    if let Some(password) = options.password.as_deref() {
        let result = handle.authenticate_password(username, password).await?;
        if result.success() {
            return Ok(());
        }
    }

    if try_authenticate_with_agent(handle, username, preferred_hash).await? {
        return Ok(());
    }

    for path in default_private_key_paths() {
        if explicit_key_paths.contains(&path) || !path.is_file() {
            continue;
        }
        let Ok(key) = load_secret_key(&path, None) else {
            continue;
        };
        let key = PrivateKeyWithHashAlg::new(Arc::new(key), preferred_hash);
        let result = handle.authenticate_publickey(username, key).await?;
        if result.success() {
            return Ok(());
        }
    }

    bail!("SSH authentication failed for user `{username}`");
}

#[cfg(unix)]
async fn try_authenticate_with_agent(
    handle: &mut client::Handle<SshClientHandler>,
    username: &str,
    preferred_hash: Option<russh::keys::ssh_key::HashAlg>,
) -> Result<bool> {
    let mut agent = match russh::keys::agent::client::AgentClient::connect_env().await {
        Ok(agent) => agent,
        Err(_) => return Ok(false),
    };
    let identities = match agent.request_identities().await {
        Ok(identities) => identities,
        Err(_) => return Ok(false),
    };

    // Keep attempts bounded to reduce server-side auth-attempt exhaustion.
    for key in identities.into_iter().take(MAX_AGENT_IDENTITIES_TO_TRY) {
        let result = handle
            .authenticate_publickey_with(username, key, preferred_hash, &mut agent)
            .await?;
        if result.success() {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(not(unix))]
async fn try_authenticate_with_agent(
    _handle: &mut client::Handle<SshClientHandler>,
    _username: &str,
    _preferred_hash: Option<russh::keys::ssh_key::HashAlg>,
) -> Result<bool> {
    Ok(false)
}

async fn detect_remote_platform(
    handle: &mut client::Handle<SshClientHandler>,
) -> Result<KaosPlatform> {
    let (os_code, os_out, os_err) = exec_capture(handle, "uname -s").await?;
    if os_code != 0 {
        bail!("Failed to detect remote OS with `uname -s`: {os_err}");
    }
    let os = normalize_os(&os_out);
    if os != "linux" && os != "macos" {
        bail!("Unsupported SSH remote OS `{os}`; only Linux and macOS targets are supported");
    }

    let (arch_code, arch_out, arch_err) = exec_capture(handle, "uname -m").await?;
    if arch_code != 0 {
        bail!("Failed to detect remote architecture with `uname -m`: {arch_err}");
    }
    let arch = normalize_arch(&arch_out);

    let libc = if os == "linux" {
        match exec_capture(handle, "ldd --version 2>&1 | head -n1").await {
            Ok((0, out, _)) => detect_libc(&out),
            _ => None,
        }
    } else {
        None
    };

    Ok(KaosPlatform {
        os,
        arch,
        abi: None,
        libc,
    })
}

async fn detect_remote_platform_with_deadline(
    handle: &mut client::Handle<SshClientHandler>,
    deadline: tokio::time::Instant,
) -> Result<KaosPlatform> {
    if tokio::time::Instant::now() >= deadline {
        bail!("Timed out before remote platform detection could start");
    }
    match tokio::time::timeout_at(deadline, detect_remote_platform(handle)).await {
        Ok(platform) => platform,
        Err(_) => bail!("Timed out while detecting remote platform"),
    }
}

async fn exec_capture(
    handle: &mut client::Handle<SshClientHandler>,
    command: &str,
) -> Result<(i32, String, String)> {
    let (code, stdout, stderr) = exec_capture_raw(handle, command).await?;
    Ok((code, stdout.trim().to_string(), stderr.trim().to_string()))
}

async fn exec_capture_raw(
    handle: &mut client::Handle<SshClientHandler>,
    command: &str,
) -> Result<(i32, String, String)> {
    let mut channel = handle.channel_open_session().await?;
    channel.exec(true, command).await?;
    loop {
        if map_channel_request_reply("execute remote command", channel.wait().await)?.is_some() {
            break;
        }
    }

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut code: Option<i32> = None;
    while let Some(msg) = channel.wait().await {
        match msg {
            ChannelMsg::Data { data } => stdout.extend_from_slice(data.as_ref()),
            ChannelMsg::ExtendedData { data, ext: 1 } => stderr.extend_from_slice(data.as_ref()),
            ChannelMsg::ExtendedData { data, .. } => stdout.extend_from_slice(data.as_ref()),
            ChannelMsg::ExitStatus { exit_status } => code = Some(exit_status as i32),
            ChannelMsg::Close => break,
            _ => {}
        }
    }

    Ok((
        code.unwrap_or(1),
        String::from_utf8_lossy(&stdout).to_string(),
        String::from_utf8_lossy(&stderr).to_string(),
    ))
}

async fn await_channel_request_reply(
    read_half: &mut russh::ChannelReadHalf,
    request_label: &str,
) -> Result<()> {
    loop {
        if map_channel_request_reply(request_label, read_half.wait().await)?.is_some() {
            return Ok(());
        }
    }
}

fn map_channel_request_reply(
    request_label: &str,
    message: Option<ChannelMsg>,
) -> Result<Option<()>> {
    match message {
        Some(ChannelMsg::Success) => Ok(Some(())),
        Some(ChannelMsg::Failure) => {
            bail!("SSH server rejected request to {request_label}")
        }
        Some(ChannelMsg::OpenFailure(reason)) => {
            bail!("SSH channel open failure while waiting to {request_label}: {reason:?}")
        }
        Some(ChannelMsg::Close) | Some(ChannelMsg::Eof) | None => {
            bail!("SSH channel closed while waiting to {request_label}")
        }
        // OpenSSH can send window-size updates before the reply to an exec/env
        // request. Those are transport-level flow-control events, not an
        // answer to the request we are waiting on, so keep polling.
        Some(ChannelMsg::WindowAdjusted { .. }) => Ok(None),
        Some(other) => bail!(
            "Unexpected SSH channel message while waiting to {request_label}: {}",
            channel_message_name(&other)
        ),
    }
}

fn channel_message_name(message: &ChannelMsg) -> &'static str {
    match message {
        ChannelMsg::Open { .. } => "open",
        ChannelMsg::Data { .. } => "data",
        ChannelMsg::ExtendedData { .. } => "extended_data",
        ChannelMsg::Eof => "eof",
        ChannelMsg::Close => "close",
        ChannelMsg::RequestPty { .. } => "request_pty",
        ChannelMsg::RequestShell { .. } => "request_shell",
        ChannelMsg::Exec { .. } => "exec",
        ChannelMsg::Signal { .. } => "signal",
        ChannelMsg::RequestSubsystem { .. } => "request_subsystem",
        ChannelMsg::RequestX11 { .. } => "request_x11",
        ChannelMsg::SetEnv { .. } => "set_env",
        ChannelMsg::WindowChange { .. } => "window_change",
        ChannelMsg::AgentForward { .. } => "agent_forward",
        ChannelMsg::XonXoff { .. } => "xon_xoff",
        ChannelMsg::ExitStatus { .. } => "exit_status",
        ChannelMsg::ExitSignal { .. } => "exit_signal",
        ChannelMsg::WindowAdjusted { .. } => "window_adjusted",
        ChannelMsg::Success => "success",
        ChannelMsg::Failure => "failure",
        ChannelMsg::OpenFailure(_) => "open_failure",
        _ => "other",
    }
}

const SSH_ENV_VAR_UNSET_EXIT_CODE: i32 = 3;

fn map_env_var_lookup_result(
    key: &str,
    exit_code: i32,
    stdout: String,
    stderr: String,
) -> Result<Option<String>> {
    match exit_code {
        0 => Ok(Some(stdout)),
        SSH_ENV_VAR_UNSET_EXIT_CODE => Ok(None),
        _ => bail!("Failed to read remote environment variable `{key}`: {stderr}"),
    }
}

async fn mkdir_once(sftp: &SftpSession, path: &str, exist_ok: bool) -> Result<()> {
    match sftp.create_dir(path).await {
        Ok(()) => Ok(()),
        Err(SftpError::Status(status))
            if exist_ok
                && matches!(
                    status.status_code,
                    StatusCode::Failure | StatusCode::NoSuchFile
                ) =>
        {
            match sftp.metadata(path).await {
                Ok(attrs) if attrs.is_dir() => Ok(()),
                Ok(_) => bail!("Path exists but is not a directory: {path}"),
                Err(err) => Err(sftp_error(
                    &KaosPath::from_style(KaosPathStyle::Posix, path),
                    "stat",
                    err,
                )),
            }
        }
        Err(err) => Err(sftp_error(
            &KaosPath::from_style(KaosPathStyle::Posix, path),
            "create directory",
            err,
        )),
    }
}

fn normalize_os(raw: &str) -> String {
    let lowered = raw.trim().to_ascii_lowercase();
    if lowered.contains("darwin") {
        "macos".to_string()
    } else if lowered.contains("linux") || lowered.is_empty() {
        "linux".to_string()
    } else {
        lowered
    }
}

fn normalize_arch(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => "x86_64".to_string(),
        "aarch64" | "arm64" => "aarch64".to_string(),
        "x86" | "i386" | "i686" => "x86".to_string(),
        other if !other.is_empty() => other.to_string(),
        _ => "x86_64".to_string(),
    }
}

fn validate_env_var_key(key: &str) -> Result<()> {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        bail!("Environment variable name cannot be empty");
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        bail!("Invalid environment variable name `{key}`");
    }
    if chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric()) {
        Ok(())
    } else {
        bail!("Invalid environment variable name `{key}`");
    }
}

fn detect_libc(raw: &str) -> Option<String> {
    let lowered = raw.to_ascii_lowercase();
    if lowered.contains("musl") {
        Some("musl".to_string())
    } else if lowered.contains("glibc") || lowered.contains("gnu libc") {
        Some("gnu".to_string())
    } else {
        None
    }
}

fn resolve_absolute_posix(base: &KaosPath, target: &str) -> String {
    if target == "~" {
        base.as_str().to_string()
    } else if let Some(stripped) = target.strip_prefix("~/") {
        Utf8TypedPathBuf::from_unix(base.as_str())
            .join(stripped)
            .normalize()
            .to_string()
    } else if target.starts_with('/') {
        Utf8TypedPathBuf::from_unix(target).normalize().to_string()
    } else {
        Utf8TypedPathBuf::from_unix(base.as_str())
            .join(target)
            .normalize()
            .to_string()
    }
}

fn normalize_glob_pattern(pattern: &str) -> String {
    pattern.trim_start_matches("./").to_string()
}

fn shell_quote(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    format!("'{}'", arg.replace('\'', r#"'"'"'"#))
}

fn sftp_error(path: &KaosPath, operation: &'static str, err: SftpError) -> anyhow::Error {
    KaosFileError::new(
        path,
        operation,
        classify_sftp_error_kind(&err),
        format!("SFTP error: {err}"),
    )
    .into()
}

fn classify_sftp_error_kind(err: &SftpError) -> KaosFileErrorKind {
    match err {
        SftpError::Status(status) => match status.status_code {
            StatusCode::NoSuchFile => KaosFileErrorKind::NotFound,
            StatusCode::PermissionDenied => KaosFileErrorKind::PermissionDenied,
            _ => KaosFileErrorKind::Other,
        },
        _ => KaosFileErrorKind::Other,
    }
}

fn expand_home(path: impl AsRef<str>) -> PathBuf {
    let path = path.as_ref();
    if path == "~"
        && let Some(home) = dirs::home_dir()
    {
        return home;
    }
    if let Some(stripped) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    PathBuf::from(path)
}

fn default_private_key_paths() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let ssh_dir = home.join(".ssh");
    [
        "id_ed25519",
        "id_ed25519_sk",
        "id_ecdsa",
        "id_ecdsa_sk",
        "id_rsa",
        "id_dsa",
    ]
    .into_iter()
    .map(|name| ssh_dir.join(name))
    .collect()
}

fn default_ssh_username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "root".to_string())
}

fn build_storage_name(host: &str, port: u16, username: &str) -> String {
    let normalized_host = host.to_ascii_lowercase();
    let host_hint = normalized_host
        .chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            _ => '-',
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let host_hint = if host_hint.is_empty() {
        "host".to_string()
    } else {
        host_hint.chars().take(24).collect()
    };

    let identity = format!("{username}@{normalized_host}:{port}");
    let mut hasher = Sha256::new();
    hasher.update(identity.as_bytes());
    let digest = hasher.finalize();
    let mut suffix = String::with_capacity(32);
    for byte in digest.iter().take(16) {
        let _ = write!(&mut suffix, "{byte:02x}");
    }

    format!("ssh_{host_hint}_{suffix}")
}

#[cfg(test)]
mod tests {
    use super::{
        GlobTraversalPlan, SSH_ENV_VAR_UNSET_EXIT_CODE, build_storage_name,
        classify_sftp_error_kind, map_channel_request_reply, map_env_var_lookup_result,
        normalize_glob_pattern, resolve_absolute_posix, validate_env_var_key,
    };
    use crate::{KaosFileErrorKind, KaosPath, KaosPathStyle};
    use russh::{ChannelMsg, ChannelOpenFailure};
    use russh_sftp::client::error::Error as SftpError;
    use russh_sftp::protocol::{Status, StatusCode};

    #[test]
    fn storage_name_is_deterministic() {
        let first = build_storage_name("a-b.example.com", 22, "ops");
        let second = build_storage_name("a-b.example.com", 22, "ops");
        assert_eq!(first, second);
    }

    #[test]
    fn storage_name_distinguishes_common_host_variants() {
        let dashed = build_storage_name("a-b.example.com", 22, "ops");
        let dotted = build_storage_name("a.b-example.com", 22, "ops");
        assert_ne!(dashed, dotted);
    }

    #[test]
    fn storage_name_ignores_host_case() {
        let lower = build_storage_name("example.com", 22, "ops");
        let mixed = build_storage_name("Example.COM", 22, "ops");
        assert_eq!(lower, mixed);
    }

    #[test]
    fn resolve_absolute_posix_expands_home_marker() {
        let home = KaosPath::from_style(KaosPathStyle::Posix, "/home/alice");
        assert_eq!(resolve_absolute_posix(&home, "~"), "/home/alice");
        assert_eq!(
            resolve_absolute_posix(&home, "~/project"),
            "/home/alice/project"
        );
    }

    #[test]
    fn glob_plan_blocks_recursion_for_shallow_patterns() {
        let plan = GlobTraversalPlan::from_pattern("*.rs").expect("valid pattern");
        let options = glob::MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: true,
        };

        assert!(!plan.should_descend("src", 1, &options));
    }

    #[test]
    fn glob_plan_keeps_prefix_pruning_with_globstar() {
        let plan = GlobTraversalPlan::from_pattern("src/**/test_*.rs").expect("valid pattern");
        let options = glob::MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: true,
        };

        assert!(plan.should_descend("src", 1, &options));
        assert!(plan.should_descend("src/lib", 2, &options));
        assert!(!plan.should_descend("docs", 1, &options));
    }

    #[test]
    fn normalize_glob_pattern_trims_current_dir_prefix() {
        assert_eq!(normalize_glob_pattern("./*.rs"), "*.rs");
        assert_eq!(normalize_glob_pattern("./src/**/*.rs"), "src/**/*.rs");
        assert_eq!(normalize_glob_pattern("*.rs"), "*.rs");
    }

    #[test]
    fn validate_env_var_key_accepts_posix_identifiers() {
        assert!(validate_env_var_key("PATH").is_ok());
        assert!(validate_env_var_key("_TOKEN_2").is_ok());
    }

    #[test]
    fn validate_env_var_key_rejects_invalid_identifiers() {
        assert!(validate_env_var_key("").is_err());
        assert!(validate_env_var_key("1PATH").is_err());
        assert!(validate_env_var_key("BAD-NAME").is_err());
    }

    #[test]
    fn map_env_var_lookup_result_returns_value_for_success() {
        let value = map_env_var_lookup_result("PATH", 0, "value ".to_string(), "noise".to_string())
            .expect("success");
        assert_eq!(value, Some("value ".to_string()));
    }

    #[test]
    fn map_env_var_lookup_result_returns_none_for_unset_exit_code() {
        let value = map_env_var_lookup_result(
            "PATH",
            SSH_ENV_VAR_UNSET_EXIT_CODE,
            String::new(),
            "startup noise".to_string(),
        )
        .expect("unset");
        assert_eq!(value, None);
    }

    #[test]
    fn map_env_var_lookup_result_errors_for_other_failures() {
        let err =
            map_env_var_lookup_result("PATH", 1, String::new(), "permission denied".to_string())
                .expect_err("failure");
        assert!(err.to_string().contains("permission denied"));
    }

    #[test]
    fn classify_sftp_error_kind_recognizes_not_found() {
        let err = SftpError::Status(Status {
            id: 1,
            status_code: StatusCode::NoSuchFile,
            error_message: "missing".to_string(),
            language_tag: String::new(),
        });
        assert_eq!(classify_sftp_error_kind(&err), KaosFileErrorKind::NotFound);
    }

    #[test]
    fn map_channel_request_reply_accepts_success() {
        let result = map_channel_request_reply("execute remote command", Some(ChannelMsg::Success))
            .expect("success");
        assert!(result.is_some());
    }

    #[test]
    fn map_channel_request_reply_ignores_window_adjusted() {
        let result = map_channel_request_reply(
            "execute remote command",
            Some(ChannelMsg::WindowAdjusted { new_size: 1024 }),
        )
        .expect("window adjusted");
        assert!(result.is_none());
    }

    #[test]
    fn map_channel_request_reply_rejects_failure() {
        let err = map_channel_request_reply("execute remote command", Some(ChannelMsg::Failure))
            .expect_err("failure");
        assert!(
            err.to_string()
                .contains("SSH server rejected request to execute remote command")
        );
    }

    #[test]
    fn map_channel_request_reply_rejects_close() {
        let err = map_channel_request_reply("execute remote command", Some(ChannelMsg::Close))
            .expect_err("close");
        assert!(
            err.to_string()
                .contains("SSH channel closed while waiting to execute remote command")
        );
    }

    #[test]
    fn map_channel_request_reply_rejects_open_failure() {
        let err = map_channel_request_reply(
            "execute remote command",
            Some(ChannelMsg::OpenFailure(
                ChannelOpenFailure::AdministrativelyProhibited,
            )),
        )
        .expect_err("open failure");
        assert!(err.to_string().contains("AdministrativelyProhibited"));
    }

    #[test]
    fn map_channel_request_reply_rejects_unexpected_message() {
        let err = map_channel_request_reply(
            "execute remote command",
            Some(ChannelMsg::ExitStatus { exit_status: 0 }),
        )
        .expect_err("unexpected message");
        assert!(err.to_string().contains("exit_status"));
    }
}
