#![cfg(feature = "boxlite-e2e")]

#[path = "support/boxlite_e2e.rs"]
mod boxlite_e2e;
#[path = "support/boxlite_mcp.rs"]
mod boxlite_mcp;
mod tool_test_utils;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use boxlite_e2e::{BoxliteSshFixture, CurrentKaosGuard, REMOTE_WORK_DIR, run_kimi_agent};
use boxlite_mcp::{
    GUEST_FIXTURE_DIR, GUEST_PYTHON, GUEST_STDIO_SCRIPT, HTTP_ENV_VALUE, HTTP_SERVER_NAME,
    HTTP_TOOL_NAME, STDIO_ENV_VALUE, STDIO_SERVER_NAME, STDIO_TOOL_NAME, provision_mcp_fixture,
};
use kaos::KaosPath;
use kosong::message::{ContentPart, TextPart, ToolCall};
use kosong::tooling::{CallableTool, ToolOutput, ToolReturnValue};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::timeout;
use tool_test_utils::RuntimeFixture;

use kimi_agent::soul::toolset::{KimiToolset, wait_mcp_loading_tasks, with_current_tool_call};

/// BoxLite-backed end-to-end coverage for the SSH Kaos MCP path.
///
/// This test is intentionally `ignored` by default because it requires all of:
/// - hardware virtualization that BoxLite can use
/// - BoxLite runtime binaries resolvable by the `boxlite` crate
/// - network access from the guest to pull an OCI base image and install `mcp[cli]~=1.26`
/// - `protoc` in the host PATH so the `boxlite-shared` build script can compile
///
/// Run it with:
/// `cargo test -p kimi-agent --features boxlite-e2e --test mcp_boxlite_ssh_e2e -- --ignored --nocapture`
///
/// For interactive guest-side debugging, add `KIMI_BOXLITE_DEBUG_TMUX=1`. That keeps long-lived
/// guest services in named tmux sessions and makes pane/log dumping much easier when the test
/// fails. The stdio MCP child still runs as a normal SSH child process because detaching it into
/// tmux would break the stdio transport semantics under test.
///
/// Coverage:
/// - CLI `mcp add/test/list/remove` against a real SSH backend
/// - host-local SQLite persistence scoped by the active SSH Kaos backend
/// - stdio MCP spawn semantics over SSH, including remote `cwd` and configured `env`
/// - HTTP MCP access to a guest-only `127.0.0.1` listener through the Kaos TCP tunnel
/// - runtime tool invocation through `KimiToolset`
/// - stdio MCP process cleanup after `KimiToolset::cleanup()`
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires BoxLite runtime binaries, virtualization, and guest network access"]
async fn mcp_over_boxlite_ssh_backend_is_end_to_end_correct() -> Result<()> {
    let fixture = provision_mcp_fixture().await?;
    let test_result = run_full_e2e(&fixture).await;
    if let Err(err) = &test_result {
        eprintln!("BoxLite SSH MCP E2E failed: {err:#}");
        fixture.dump_debug_artifacts().await;
    }
    let shutdown_result = fixture.shutdown().await;

    test_result?;
    shutdown_result?;
    Ok(())
}

async fn run_full_e2e(fixture: &BoxliteSshFixture) -> Result<()> {
    exercise_cli_roundtrip(fixture).await?;
    exercise_runtime_tool_invocation(fixture).await?;
    Ok(())
}

async fn exercise_cli_roundtrip(fixture: &BoxliteSshFixture) -> Result<()> {
    // The host must not be able to bypass the SSH tunnel and connect to the guest-only MCP HTTP
    // listener directly. If this assertion fails, the test is no longer validating Kaos tunnel
    // semantics.
    let host_http_probe = std::net::TcpStream::connect(("127.0.0.1", fixture.guest_http_port()));
    assert!(
        host_http_probe.is_err(),
        "guest-only HTTP port should not be reachable from the host"
    );

    assert!(
        !fixture.local_state_db_path().exists(),
        "SQLite state should not exist before the first MCP mutation"
    );
    assert!(
        !fixture.local_legacy_mcp_config_path().exists(),
        "legacy local mcp.json should not be written"
    );

    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &[
            "mcp",
            "add",
            STDIO_SERVER_NAME,
            "--transport",
            "stdio",
            "--env",
            &format!("BOX_MCP_ENV={STDIO_ENV_VALUE}"),
            "--env",
            "FIXTURE_TRANSPORT=stdio",
            "--",
            GUEST_PYTHON,
            GUEST_STDIO_SCRIPT,
        ],
    )?;

    let host_config = fixture.read_host_mcp_config()?;
    let stdio_server = server_config(&host_config, STDIO_SERVER_NAME)?;
    assert_eq!(stdio_server["command"], GUEST_PYTHON);
    assert_eq!(stdio_server["args"], json!([GUEST_STDIO_SCRIPT]));
    assert_eq!(stdio_server["env"]["BOX_MCP_ENV"], STDIO_ENV_VALUE);
    assert_eq!(stdio_server["env"]["FIXTURE_TRANSPORT"], json!("stdio"));
    assert!(
        !fixture.remote_legacy_mcp_config_exists().await?,
        "legacy remote mcp.json should not be written when SQLite is authoritative"
    );

    let stdio_test = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "test", STDIO_SERVER_NAME],
    )?;
    assert!(stdio_test.stdout.contains(STDIO_TOOL_NAME));

    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &[
            "mcp",
            "add",
            HTTP_SERVER_NAME,
            "--transport",
            "http",
            &format!("http://127.0.0.1:{}/mcp", fixture.guest_http_port()),
        ],
    )?;

    let host_config = fixture.read_host_mcp_config()?;
    let http_server = server_config(&host_config, HTTP_SERVER_NAME)?;
    assert_eq!(
        http_server["url"],
        json!(format!(
            "http://127.0.0.1:{}/mcp",
            fixture.guest_http_port()
        ))
    );
    assert_eq!(http_server["transport"], "http");

    let http_test = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "test", HTTP_SERVER_NAME],
    )?;
    assert!(http_test.stdout.contains(HTTP_TOOL_NAME));

    let list_output = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "list"],
    )?;
    assert!(list_output.stdout.contains(STDIO_SERVER_NAME));
    assert!(list_output.stdout.contains(HTTP_SERVER_NAME));

    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "remove", STDIO_SERVER_NAME],
    )?;
    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "remove", HTTP_SERVER_NAME],
    )?;

    let host_config = fixture.read_host_mcp_config()?;
    let servers = host_config["mcpServers"]
        .as_object()
        .context("mcpServers should be an object")?;
    assert!(
        !servers.contains_key(STDIO_SERVER_NAME),
        "removed stdio server should disappear from SQLite-backed MCP config"
    );
    assert!(
        !servers.contains_key(HTTP_SERVER_NAME),
        "removed HTTP server should disappear from SQLite-backed MCP config"
    );
    assert_eq!(
        servers.len(),
        0,
        "SQLite-backed MCP config should be empty after cleanup"
    );

    let list_output = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "list"],
    )?;
    assert!(list_output.stdout.contains("No MCP servers configured."));

    Ok(())
}

async fn exercise_runtime_tool_invocation(fixture: &BoxliteSshFixture) -> Result<()> {
    const RUNTIME_TIMEOUT: Duration = Duration::from_secs(30);

    let ssh_kaos = Arc::new(fixture.connect_ssh_kaos().await?);
    let remote_work_dir = KaosPath::new(REMOTE_WORK_DIR);

    let _guard = CurrentKaosGuard::new(ssh_kaos);

    let mut runtime_fixture = RuntimeFixture::new();
    runtime_fixture.runtime.session.work_dir = remote_work_dir.clone();
    runtime_fixture.runtime.builtin_args.KIMI_WORK_DIR = remote_work_dir;

    let toolset = Arc::new(tokio::sync::Mutex::new(KimiToolset::new()));
    let runtime_config = json!({
        "mcpServers": {
            STDIO_SERVER_NAME: {
                "command": GUEST_PYTHON,
                "args": [GUEST_STDIO_SCRIPT],
                "env": {
                    "BOX_MCP_ENV": STDIO_ENV_VALUE,
                    "FIXTURE_TRANSPORT": "stdio"
                }
            },
            HTTP_SERVER_NAME: {
                "url": format!("http://127.0.0.1:{}/mcp", fixture.guest_http_port()),
                "transport": "http"
            }
        }
    });

    {
        let mut guard = toolset.lock().await;
        timeout(
            RUNTIME_TIMEOUT,
            guard.load_mcp_tools(
                &[runtime_config],
                &runtime_fixture.runtime,
                Arc::clone(&toolset),
            ),
        )
        .await
        .context("timed out while loading MCP tools through runtime")?
        .context("load MCP tools through runtime")?;
    }

    let loading_tasks = {
        let mut guard = toolset.lock().await;
        guard.take_mcp_loading_tasks()
    };
    timeout(RUNTIME_TIMEOUT, wait_mcp_loading_tasks(loading_tasks))
        .await
        .context("timed out while waiting for runtime MCP tools")?
        .context("wait for runtime MCP tools")?;

    {
        let mut guard = toolset.lock().await;
        let stdio_tool = guard
            .find(STDIO_TOOL_NAME)
            .context("missing stdio fixture tool")?;
        let http_tool = guard
            .find(HTTP_TOOL_NAME)
            .context("missing HTTP fixture tool")?;

        let stdio_context = parse_context_from_tool(stdio_tool.as_ref(), json!({}))
            .await
            .context("call stdio fixture tool")?;
        let http_context = parse_context_from_tool(http_tool.as_ref(), json!({}))
            .await
            .context("call HTTP fixture tool")?;

        assert_eq!(stdio_context.transport, "stdio");
        assert_eq!(stdio_context.cwd, REMOTE_WORK_DIR);
        assert_eq!(stdio_context.env_value.as_deref(), Some(STDIO_ENV_VALUE));
        assert!(stdio_context.pid > 0);

        assert_eq!(http_context.transport, "http");
        assert_eq!(http_context.cwd, GUEST_FIXTURE_DIR);
        assert_eq!(http_context.env_value.as_deref(), Some(HTTP_ENV_VALUE));
        assert!(http_context.pid > 0);
        assert_ne!(stdio_context.pid, http_context.pid);
        assert_eq!(stdio_context.hostname, http_context.hostname);

        timeout(RUNTIME_TIMEOUT, guard.cleanup())
            .await
            .context("timed out while cleaning up runtime MCP tools")?;
        assert!(guard.find(STDIO_TOOL_NAME).is_none());
        assert!(guard.find(HTTP_TOOL_NAME).is_none());

        fixture
            .wait_for_remote_process_exit(stdio_context.pid)
            .await
            .context("stdio MCP process should be reaped by cleanup")?;
        assert!(
            fixture
                .remote_process_exists(http_context.pid)
                .await
                .context("HTTP fixture should still be running")?,
            "toolset cleanup should not kill the external HTTP fixture process"
        );
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct FixtureContext {
    transport: String,
    cwd: String,
    env_value: Option<String>,
    pid: u32,
    hostname: String,
}

async fn parse_context_from_tool(
    tool: &dyn CallableTool,
    arguments: Value,
) -> Result<FixtureContext> {
    let tool_call = ToolCall::new("boxlite-e2e-tool-call", &tool.base().name);
    let result = with_current_tool_call(tool_call, tool.call(arguments)).await;
    if result.is_error {
        bail!("tool returned error: {:?}", result.message);
    }

    let text = extract_text_output(&result)?;
    let context: FixtureContext = serde_json::from_str(&text).context("parse fixture context")?;
    assert!(!context.hostname.trim().is_empty());
    Ok(context)
}

fn extract_text_output(result: &ToolReturnValue) -> Result<String> {
    match &result.output {
        ToolOutput::Text(text) => Ok(text.clone()),
        ToolOutput::Parts(parts) => {
            let mut text = String::new();
            for part in parts {
                if let ContentPart::Text(TextPart { text: chunk, .. }) = part {
                    text.push_str(chunk);
                }
            }
            if text.is_empty() {
                bail!("tool returned no text content");
            }
            Ok(text)
        }
    }
}

fn server_config<'a>(config: &'a Value, name: &str) -> Result<&'a Value> {
    config["mcpServers"][name]
        .as_object()
        .map(|_| &config["mcpServers"][name])
        .with_context(|| format!("missing MCP server config for {name}"))
}
