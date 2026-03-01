#![cfg(feature = "boxlite-e2e")]

#[path = "support/boxlite_e2e.rs"]
mod boxlite_e2e;
#[path = "support/oauth_mock.rs"]
mod oauth_mock;
mod tool_test_utils;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use boxlite_e2e::{
    BoxliteSshFixture, CurrentKaosGuard, REMOTE_WORK_DIR, run_kimi_agent, run_kimi_agent_with_env,
};
use kaos::{KaosPath, get_current_kaos};
use kosong::message::{ContentPart, TextPart, ToolCall};
use kosong::tooling::{CallableTool, ToolOutput, ToolReturnValue};
use oauth_mock::{
    HostOauthFixture, OAuthFixtureState, assert_remote_auth_file_contains_token,
    provision_oauth_fixture,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::timeout;
use tool_test_utils::RuntimeFixture;

use kimi_agent::soul::toolset::{KimiToolset, wait_mcp_loading_tasks, with_current_tool_call};

/// BoxLite-backed OAuth end-to-end coverage for HTTP MCP over SSH Kaos.
///
/// This test keeps the entire protected OAuth MCP stack inside the guest. The CLI and runtime only
/// know the guest-local MCP URL, so HTTP access still has to flow through the SSH Kaos TCP tunnel.
/// The only host-visible surface is a forwarded browser authorization port used to complete the
/// local OAuth callback step.
///
/// Run it with:
/// `KIMI_BOXLITE_DEBUG_TMUX=1 cargo test -p kimi-agent --features boxlite-e2e --test mcp_boxlite_ssh_oauth_e2e -- --ignored --nocapture`
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires BoxLite runtime binaries, virtualization, guest network access, and OAuth browser helper execution"]
async fn oauth_http_mcp_over_boxlite_ssh_backend_is_end_to_end_correct() -> Result<()> {
    let fixture = provision_oauth_fixture().await?;
    let oauth = HostOauthFixture::start(&fixture).await?;
    let test_result = run_oauth_e2e(&fixture, &oauth).await;
    if let Err(err) = &test_result {
        eprintln!("BoxLite SSH OAuth MCP E2E failed: {err:#}");
        oauth.dump_debug_artifacts(&fixture).await;
        fixture.dump_debug_artifacts().await;
    }
    let oauth_shutdown = oauth.shutdown().await;
    let fixture_shutdown = fixture.shutdown().await;

    test_result?;
    oauth_shutdown?;
    fixture_shutdown?;
    Ok(())
}

async fn run_oauth_e2e(fixture: &BoxliteSshFixture, oauth: &HostOauthFixture) -> Result<()> {
    assert_guest_oauth_fixture_is_ready(fixture).await?;
    exercise_cli_oauth_roundtrip(fixture, oauth).await?;
    exercise_runtime_oauth_tool_invocation(fixture, oauth).await?;
    exercise_cli_reset_auth(fixture, oauth).await?;
    Ok(())
}

async fn assert_guest_oauth_fixture_is_ready(fixture: &BoxliteSshFixture) -> Result<()> {
    let initial_state = fixture.read_guest_oauth_state().await?;
    let parsed: OAuthFixtureState =
        serde_json::from_str(&initial_state).context("parse initial guest OAuth state")?;
    assert_eq!(parsed.authorized_mcp_requests, 0);
    assert_eq!(parsed.register_requests, 0);
    assert_eq!(parsed.authorize_requests, 0);
    assert_eq!(parsed.token_requests, 0);
    Ok(())
}

async fn exercise_cli_oauth_roundtrip(
    fixture: &BoxliteSshFixture,
    oauth: &HostOauthFixture,
) -> Result<()> {
    let server_name = oauth.server_name();
    let remote_mcp_url = oauth.remote_mcp_url();

    assert!(
        !fixture.local_mcp_auth_path().exists(),
        "local OAuth cache should stay empty when Kaos backend is SSH"
    );

    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &[
            "mcp",
            "add",
            server_name,
            "--transport",
            "http",
            "--auth",
            "oauth",
            &remote_mcp_url,
        ],
    )?;

    let list_before_auth = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "list"],
    )?;
    assert!(list_before_auth.stdout.contains(server_name));
    assert!(list_before_auth.stdout.contains("authorization required"));

    let browser_script = oauth.write_browser_script(fixture.host_home()).await?;
    let browser_script_path = browser_script.to_string_lossy().into_owned();
    let remote_authority = oauth.remote_authority().to_string();
    let local_authority = oauth.local_authority().to_string();

    let auth_output = run_kimi_agent_with_env(
        fixture.host_home(),
        fixture.cli_config_path(),
        &[
            ("BROWSER", browser_script_path.as_str()),
            (
                "KIMI_TEST_BROWSER_REMOTE_AUTHORITY",
                remote_authority.as_str(),
            ),
            (
                "KIMI_TEST_BROWSER_LOCAL_AUTHORITY",
                local_authority.as_str(),
            ),
        ],
        &["mcp", "auth", server_name],
    )?;
    assert!(auth_output.stdout.contains("Successfully authorized"));
    assert!(auth_output.stdout.contains("Available tools: 1"));

    let remote_auth = fixture.read_remote_mcp_auth_file().await?;
    assert_remote_auth_file_contains_token(&remote_auth, &remote_mcp_url)?;
    assert!(
        !fixture.local_mcp_auth_path().exists(),
        "local OAuth cache should remain empty after authorization"
    );
    let state_after_auth = oauth.read_state(fixture).await?;
    assert_completed_oauth_handshake(&state_after_auth);
    assert_eq!(
        state_after_auth.last_authorization_header.as_deref(),
        Some("Bearer boxlite-oauth-access-token")
    );
    assert!(
        state_after_auth.authorized_mcp_requests >= 1,
        "auth flow should reach the protected MCP resource at least once"
    );

    let list_after_auth = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "list"],
    )?;
    assert!(list_after_auth.stdout.contains(server_name));
    assert!(!list_after_auth.stdout.contains("authorization required"));

    let test_output = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "test", server_name],
    )?;
    assert!(test_output.stdout.contains(oauth.tool_name()));
    let state_after_test = oauth.read_state(fixture).await?;
    assert!(
        state_after_test.authorized_mcp_requests > state_after_auth.authorized_mcp_requests,
        "mcp test should perform an additional authorized MCP request"
    );

    Ok(())
}

async fn exercise_runtime_oauth_tool_invocation(
    fixture: &BoxliteSshFixture,
    oauth: &HostOauthFixture,
) -> Result<()> {
    const RUNTIME_TIMEOUT: Duration = Duration::from_secs(30);

    let ssh_kaos = Arc::new(fixture.connect_ssh_kaos().await?);
    let remote_work_dir = KaosPath::new(REMOTE_WORK_DIR);

    let _guard = CurrentKaosGuard::new(ssh_kaos);

    let mut runtime_fixture = RuntimeFixture::new();
    runtime_fixture.runtime.session.work_dir = remote_work_dir.clone();
    runtime_fixture.runtime.session.work_dir_meta.path = remote_work_dir.to_string_lossy();
    runtime_fixture.runtime.session.work_dir_meta.kaos = get_current_kaos().storage_name();
    runtime_fixture.runtime.builtin_args.KIMI_WORK_DIR = remote_work_dir;

    let toolset = Arc::new(tokio::sync::Mutex::new(KimiToolset::new()));
    let state_before_runtime = oauth.read_state(fixture).await?;
    let runtime_config = json!({
        "mcpServers": {
            oauth.server_name(): {
                "url": oauth.remote_mcp_url(),
                "transport": "http",
                "auth": "oauth",
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
        .context("timed out while loading OAuth MCP tools through runtime")?
        .context("load OAuth MCP tools through runtime")?;
    }

    let loading_tasks = {
        let mut guard = toolset.lock().await;
        guard.take_mcp_loading_tasks()
    };
    timeout(RUNTIME_TIMEOUT, wait_mcp_loading_tasks(loading_tasks))
        .await
        .context("timed out while waiting for OAuth MCP tools")?
        .context("wait for OAuth MCP tools")?;

    {
        let mut guard = toolset.lock().await;
        let oauth_tool = guard
            .find(oauth.tool_name())
            .context("missing OAuth MCP fixture tool")?;

        let context = parse_context_from_tool(oauth_tool.as_ref(), json!({}))
            .await
            .context("call OAuth MCP fixture tool")?;
        assert_eq!(context.transport, "oauth-http");
        assert_eq!(
            context.last_authorization_header.as_deref(),
            Some("Bearer boxlite-oauth-access-token")
        );

        timeout(RUNTIME_TIMEOUT, guard.cleanup())
            .await
            .context("timed out while cleaning up OAuth MCP tools")?;
        assert!(guard.find(oauth.tool_name()).is_none());
    }

    assert_eq!(
        oauth.last_authorization_header(fixture).await?.as_deref(),
        Some("Bearer boxlite-oauth-access-token")
    );
    let state_after_runtime = oauth.read_state(fixture).await?;
    assert!(
        state_after_runtime.authorized_mcp_requests > state_before_runtime.authorized_mcp_requests,
        "runtime MCP tool call should perform an additional authorized request"
    );

    Ok(())
}

async fn exercise_cli_reset_auth(
    fixture: &BoxliteSshFixture,
    oauth: &HostOauthFixture,
) -> Result<()> {
    run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "reset-auth", oauth.server_name()],
    )?;

    let remote_auth = fixture.read_remote_mcp_auth_file().await?;
    assert!(
        !remote_auth.contains(&oauth.remote_mcp_url()),
        "reset-auth should remove the remote OAuth credential entry"
    );

    let list_after_reset = run_kimi_agent(
        fixture.host_home(),
        fixture.cli_config_path(),
        &["mcp", "list"],
    )?;
    assert!(list_after_reset.stdout.contains("authorization required"));

    Ok(())
}

fn assert_completed_oauth_handshake(state: &OAuthFixtureState) {
    assert!(
        state.resource_metadata_requests >= 1,
        "OAuth flow should request protected resource metadata"
    );
    assert!(
        state.authorization_metadata_requests >= 1,
        "OAuth flow should request authorization server metadata"
    );
    assert!(
        state.register_requests >= 1,
        "OAuth flow should dynamically register a client"
    );
    assert!(
        state.authorize_requests >= 1,
        "OAuth flow should hit the authorization endpoint"
    );
    assert!(
        state.token_requests >= 1,
        "OAuth flow should exchange the authorization code for a token"
    );
}

#[derive(Debug, Deserialize)]
struct OAuthToolContext {
    transport: String,
    last_authorization_header: Option<String>,
}

async fn parse_context_from_tool(
    tool: &dyn CallableTool,
    arguments: Value,
) -> Result<OAuthToolContext> {
    let tool_call = ToolCall::new("boxlite-oauth-tool-call", &tool.base().name);
    let result = with_current_tool_call(tool_call, tool.call(arguments)).await;
    if result.is_error {
        bail!("tool returned error: {:?}", result.message);
    }

    let text = extract_text_output(&result)?;
    serde_json::from_str(&text).context("parse OAuth tool context")
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
