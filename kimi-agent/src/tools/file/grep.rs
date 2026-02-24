use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use futures::StreamExt;
use kaos::{AsyncReadable, KaosPath, KaosPlatform};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{debug, error, info};

use kosong::tooling::{CallableTool2, ToolReturnValue, tool_error};

use crate::soul::agent::Runtime;
use crate::tools::utils::ToolResultBuilder;

use super::GREP_DESC;

const RG_VERSION: &str = "15.0.0";
const RG_BASE_URL: &str = "http://cdn.kimi.com/binaries/kimi-cli/rg";

static RG_DOWNLOAD_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepParams {
    #[schemars(description = "The regular expression pattern to search for in file contents")]
    pub pattern: String,
    #[serde(default = "default_grep_path")]
    #[schemars(
        description = "File or directory to search in. Defaults to current working directory. If specified, it must be an absolute path.",
        default = "default_grep_path"
    )]
    pub path: String,
    #[serde(default)]
    #[schemars(
        description = "Glob pattern to filter files (e.g. `*.js`, `*.{ts,tsx}`). No filter by default."
    )]
    pub glob: Option<String>,
    #[serde(default = "default_output_mode")]
    #[schemars(
        description = "`content`: Show matching lines (supports `-B`, `-A`, `-C`, `-n`, `head_limit`); `files_with_matches`: Show file paths (supports `head_limit`); `count_matches`: Show total number of matches. Defaults to `files_with_matches`.",
        default = "default_output_mode"
    )]
    pub output_mode: String,
    #[serde(default, rename = "-B")]
    #[schemars(
        description = "Number of lines to show before each match (the `-B` option). Requires `output_mode` to be `content`."
    )]
    pub before_context: Option<i64>,
    #[serde(default, rename = "-A")]
    #[schemars(
        description = "Number of lines to show after each match (the `-A` option). Requires `output_mode` to be `content`."
    )]
    pub after_context: Option<i64>,
    #[serde(default, rename = "-C")]
    #[schemars(
        description = "Number of lines to show before and after each match (the `-C` option). Requires `output_mode` to be `content`."
    )]
    pub context: Option<i64>,
    #[serde(default, rename = "-n")]
    #[schemars(
        description = "Show line numbers in output (the `-n` option). Requires `output_mode` to be `content`."
    )]
    pub line_number: bool,
    #[serde(default, rename = "-i")]
    #[schemars(description = "Case insensitive search (the `-i` option).")]
    pub ignore_case: bool,
    #[serde(default, rename = "type")]
    #[schemars(
        description = "File type to search. Examples: py, rust, js, ts, go, java, etc. More efficient than `glob` for standard file types."
    )]
    pub file_type: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Limit output to first N lines, equivalent to `| head -N`. Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count_matches (limits count entries). By default, no limit is applied."
    )]
    pub head_limit: Option<i64>,
    #[serde(default)]
    #[schemars(
        description = "Enable multiline mode where `.` matches newlines and patterns can span lines (the `-U` and `--multiline-dotall` options). By default, multiline mode is disabled."
    )]
    pub multiline: bool,
}

fn default_grep_path() -> String {
    ".".to_string()
}

fn default_output_mode() -> String {
    "files_with_matches".to_string()
}

pub struct Grep {
    description: String,
}

impl Grep {
    pub fn new(_runtime: &Runtime) -> Self {
        Self {
            description: GREP_DESC.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl CallableTool2 for Grep {
    type Params = GrepParams;

    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn call_typed(&self, params: Self::Params) -> ToolReturnValue {
        let mut builder = ToolResultBuilder::default();
        let rg_command = match ensure_rg_command().await {
            Ok(command) => command,
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to locate ripgrep binary. Error: {err}"),
                    "Failed to grep",
                );
            }
        };

        let mut args = vec![rg_command];
        if params.ignore_case {
            args.push("-i".to_string());
        }
        if params.multiline {
            args.push("-U".to_string());
            args.push("--multiline-dotall".to_string());
        }
        if params.output_mode == "content" {
            if let Some(before) = params.before_context {
                args.push("-B".to_string());
                args.push(before.to_string());
            }
            if let Some(after) = params.after_context {
                args.push("-A".to_string());
                args.push(after.to_string());
            }
            if let Some(context) = params.context {
                args.push("-C".to_string());
                args.push(context.to_string());
            }
            if params.line_number {
                args.push("-n".to_string());
            }
        }
        if let Some(glob) = &params.glob {
            args.push("-g".to_string());
            args.push(glob.clone());
        }
        if let Some(file_type) = &params.file_type {
            args.push("--type".to_string());
            args.push(file_type.clone());
        }

        if params.output_mode == "files_with_matches" {
            args.push("-l".to_string());
        } else if params.output_mode == "count_matches" {
            args.push("-c".to_string());
        }

        if params.pattern.starts_with('-') {
            args.push("--".to_string());
        }
        args.push(params.pattern.clone());
        args.push(params.path.clone());

        let mut process = match kaos::exec(&args).await {
            Ok(process) => process,
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to grep. Error: {err}"),
                    "Failed to grep",
                );
            }
        };
        if let Err(err) = process.stdin().close().await {
            return tool_error(
                "",
                format!("Failed to grep. Error: {err}"),
                "Failed to grep",
            );
        }

        let (stdout_bytes, stderr_bytes) = match (process.take_stdout(), process.take_stderr()) {
            (Some(stdout), Some(stderr)) => {
                let stdout_fut = read_stream_to_end(stdout);
                let stderr_fut = read_stream_to_end(stderr);
                match tokio::try_join!(stdout_fut, stderr_fut) {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                }
            }
            (Some(stdout), None) => {
                let stdout = match read_stream_to_end(stdout).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                let stderr = match read_stream_to_end_ref(process.stderr()).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                (stdout, stderr)
            }
            (None, Some(stderr)) => {
                let stdout = match read_stream_to_end_ref(process.stdout()).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                let stderr = match read_stream_to_end(stderr).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                (stdout, stderr)
            }
            (None, None) => {
                let stdout = match read_stream_to_end_ref(process.stdout()).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                let stderr = match read_stream_to_end_ref(process.stderr()).await {
                    Ok(output) => output,
                    Err(err) => {
                        return tool_error(
                            "",
                            format!("Failed to grep. Error: {err}"),
                            "Failed to grep",
                        );
                    }
                };
                (stdout, stderr)
            }
        };

        let exitcode = match process.wait().await {
            Ok(code) => code,
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to grep. Error: {err}"),
                    "Failed to grep",
                );
            }
        };

        if exitcode != 0 && exitcode != 1 {
            let stderr = String::from_utf8_lossy(&stderr_bytes);
            let message = if stderr.trim().is_empty() {
                format!("Failed to grep. Exit status: {exitcode}")
            } else {
                format!("Failed to grep. Error: {stderr}")
            };
            return tool_error("", message, "Failed to grep");
        }

        let mut output_text = String::from_utf8_lossy(&stdout_bytes).to_string();
        if output_text.is_empty() {
            return builder.ok("No matches found", "");
        }

        let mut message = String::new();
        if let Some(limit) = params.head_limit {
            let limit = limit.max(0) as usize;
            let lines: Vec<&str> = output_text.split('\n').collect();
            if lines.len() > limit {
                let mut truncated = lines[..limit].join("\n");
                truncated.push_str(&format!("\n... (results truncated to {limit} lines)"));
                output_text = truncated;
                message = format!("Results truncated to first {limit} lines");
            }
        }

        builder.write(&output_text);
        builder.ok(&message, "")
    }
}

async fn read_stream_to_end(mut stream: Box<dyn AsyncReadable>) -> Result<Vec<u8>, String> {
    read_stream_to_end_ref(stream.as_mut()).await
}

async fn read_stream_to_end_ref(stream: &mut dyn AsyncReadable) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    loop {
        let chunk = stream
            .read(8192)
            .await
            .map_err(|err| format!("Failed to read process stream: {err}"))?;
        if chunk.is_empty() {
            break;
        }
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

fn rg_download_lock() -> &'static tokio::sync::Mutex<()> {
    RG_DOWNLOAD_LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn rg_binary_name(platform: &KaosPlatform) -> &'static str {
    if platform.os == "windows" {
        "rg.exe"
    } else {
        "rg"
    }
}

async fn find_existing_rg(bin_name: &str) -> Option<String> {
    let share_bin = kaos_share_bin_dir().joinpath(bin_name);
    if share_bin.is_file(true).await {
        return Some(share_bin.to_string_lossy());
    }

    // Only local backend can access bundled deps beside the executable.
    if current_kaos_is_local()
        && let Some(local_dep) = find_local_dep(bin_name)
    {
        return Some(local_dep.to_string_lossy().to_string());
    }

    if command_on_backend_available(bin_name).await {
        return Some(bin_name.to_string());
    }

    None
}

fn current_kaos_is_local() -> bool {
    kaos::get_current_kaos().name() == "local"
}

fn kaos_share_bin_dir() -> KaosPath {
    // KIMI_SHARE_DIR is a host-side override and should only affect local backend.
    if current_kaos_is_local()
        && let Some(path) = std::env::var_os("KIMI_SHARE_DIR")
        && !path.is_empty()
    {
        return KaosPath::from(PathBuf::from(path)).joinpath("bin");
    }
    KaosPath::home().joinpath(".kimi").joinpath("bin")
}

async fn command_on_backend_available(command: &str) -> bool {
    let args = vec![command.to_string(), "--version".to_string()];
    let mut process = match kaos::exec(&args).await {
        Ok(process) => process,
        Err(_) => return false,
    };
    matches!(process.wait().await, Ok(0))
}

fn find_local_dep(bin_name: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local_dep = manifest_dir
        .join("src")
        .join("deps")
        .join("bin")
        .join(bin_name);
    if local_dep.is_file() {
        return Some(local_dep);
    }

    let exe_dep = std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .map(|parent| parent.join("deps").join("bin").join(bin_name))
    });
    if let Some(path) = exe_dep
        && path.is_file()
    {
        return Some(path);
    }

    None
}

fn detect_target(platform: &KaosPlatform) -> Option<String> {
    let arch = match platform.arch.as_str() {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "aarch64",
        other => {
            error!("Unsupported architecture for ripgrep: {}", other);
            return None;
        }
    };

    let os = match platform.os.as_str() {
        "macos" => "apple-darwin",
        "linux" => {
            if arch == "x86_64" || platform.libc.as_deref() == Some("musl") {
                "unknown-linux-musl"
            } else {
                "unknown-linux-gnu"
            }
        }
        "windows" => "pc-windows-msvc",
        other => {
            error!("Unsupported operating system for ripgrep: {}", other);
            return None;
        }
    };

    Some(format!("{arch}-{os}"))
}

async fn ensure_rg_command() -> Result<String, String> {
    let platform = kaos::platform();
    let bin_name = rg_binary_name(&platform);
    if let Some(existing) = find_existing_rg(bin_name).await {
        debug!("Using ripgrep binary: {}", existing);
        return Ok(existing);
    }

    let _guard = rg_download_lock().lock().await;
    if let Some(existing) = find_existing_rg(bin_name).await {
        debug!("Using ripgrep binary: {}", existing);
        return Ok(existing);
    }

    download_and_install_rg(bin_name, &platform).await
}

async fn download_and_install_rg(
    bin_name: &str,
    platform: &KaosPlatform,
) -> Result<String, String> {
    let target = detect_target(platform)
        .ok_or_else(|| "Unsupported platform for ripgrep download".to_string())?;
    let is_windows = target.contains("windows");
    let archive_ext = if is_windows { "zip" } else { "tar.gz" };
    let filename = format!("ripgrep-{RG_VERSION}-{target}.{archive_ext}");
    let url = format!("{RG_BASE_URL}/{filename}");
    info!("Downloading ripgrep from {}", url);

    let response = reqwest::get(&url)
        .await
        .map_err(|err| format!("Failed to download ripgrep: {err}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to download ripgrep: HTTP {}",
            response.status()
        ));
    }

    let mut archive_bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| format!("Failed to download ripgrep: {err}"))?;
        archive_bytes.extend_from_slice(&chunk);
    }

    let share_bin = kaos_share_bin_dir();
    kaos::mkdir(&share_bin, true, true)
        .await
        .map_err(|err| format!("Failed to create share bin dir: {err}"))?;
    let destination = share_bin.joinpath(bin_name);

    let bin_name_owned = bin_name.to_string();
    let bin_bytes = tokio::task::spawn_blocking(move || {
        if is_windows {
            extract_zip_bytes(&archive_bytes, &bin_name_owned)
        } else {
            extract_tar_bytes(&archive_bytes, &bin_name_owned)
        }
    })
    .await
    .map_err(|err| format!("Failed to extract ripgrep: {err}"))?
    .map_err(|err| format!("Failed to extract ripgrep: {err}"))?;

    destination
        .write_bytes(&bin_bytes)
        .await
        .map_err(|err| format!("Failed to write ripgrep binary: {err}"))?;

    kaos::chmod(&destination, 0o755)
        .await
        .map_err(|err| format!("Failed to set permissions: {err}"))?;

    info!("Installed ripgrep to {}", destination);

    Ok(destination.to_string_lossy())
}

fn extract_zip_bytes(archive_bytes: &[u8], bin_name: &str) -> Result<Vec<u8>, String> {
    let reader = std::io::Cursor::new(archive_bytes);
    let mut archive =
        zip::ZipArchive::new(reader).map_err(|err| format!("Failed to read zip archive: {err}"))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|err| format!("Failed to read zip entry: {err}"))?;
        let entry_name = entry.name();
        if Path::new(entry_name)
            .file_name()
            .and_then(|name| name.to_str())
            == Some(bin_name)
        {
            let mut buf = Vec::new();
            std::io::copy(&mut entry, &mut buf)
                .map_err(|err| format!("Failed to extract ripgrep: {err}"))?;
            return Ok(buf);
        }
    }

    Err("Ripgrep binary not found in archive".to_string())
}

fn extract_tar_bytes(archive_bytes: &[u8], bin_name: &str) -> Result<Vec<u8>, String> {
    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(archive_bytes));
    let mut archive = tar::Archive::new(decoder);

    let entries = archive
        .entries()
        .map_err(|err| format!("Failed to read tar archive: {err}"))?;
    for entry in entries {
        let mut entry = entry.map_err(|err| format!("Failed to read tar entry: {err}"))?;
        let path = entry
            .path()
            .map_err(|err| format!("Failed to read tar entry path: {err}"))?;
        if path.file_name().and_then(|name| name.to_str()) == Some(bin_name) {
            let mut buf = Vec::new();
            std::io::copy(&mut entry, &mut buf)
                .map_err(|err| format!("Failed to extract ripgrep: {err}"))?;
            return Ok(buf);
        }
    }

    Err("Ripgrep binary not found in archive".to_string())
}
