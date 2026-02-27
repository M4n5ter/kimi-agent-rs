use std::path::Path;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::error::Error as WsError;

use kaos::KaosPath;
use kimi_agent::config::{Config, LLMModel, LLMProvider, ProviderType, get_default_config};
use kimi_agent::constant::{NAME, VERSION};
use kimi_agent::session::Session;
use kimi_agent::wire::protocol::WIRE_PROTOCOL_VERSION;
use kimi_agent::wire::server::{WireWsServer, WsSessionRuntimeOptions};

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

fn set_home_env(path: &Path) -> Vec<EnvGuard> {
    let share_dir = path.join(".kimi");
    vec![
        EnvGuard::set("HOME", path.to_str().expect("home path")),
        EnvGuard::set("USERPROFILE", path.to_str().expect("home path")),
        EnvGuard::set(
            "KIMI_SHARE_DIR",
            share_dir.to_str().expect("share dir path"),
        ),
    ]
}

fn scripted_config() -> Config {
    let mut config = get_default_config();
    config.default_model = "scripted".to_string();
    config.models.insert(
        "scripted".to_string(),
        LLMModel {
            provider: "scripted_provider".to_string(),
            model: "scripted_echo".to_string(),
            max_context_size: 100_000,
            capabilities: None,
        },
    );
    config.providers.insert(
        "scripted_provider".to_string(),
        LLMProvider {
            provider_type: ProviderType::ScriptedEcho,
            base_url: String::new(),
            api_key: String::new(),
            env: None,
            custom_headers: None,
        },
    );
    config
}

fn configure_scripted_env(home_dir: &TempDir, scripts: &[&str]) -> Vec<EnvGuard> {
    let mut env = set_home_env(home_dir.path());
    let scripts_path = home_dir.path().join("scripts.json");
    let scripts_json: Vec<String> = scripts.iter().map(|script| script.to_string()).collect();
    std::fs::write(
        &scripts_path,
        serde_json::to_string(&scripts_json).expect("serialize scripts"),
    )
    .expect("write scripted echo file");
    env.push(EnvGuard::set(
        "KIMI_SCRIPTED_ECHO_SCRIPTS",
        scripts_path.to_str().expect("scripts path"),
    ));
    env
}

async fn connect_ws_with_retry(
    url: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    let mut last_err = String::new();
    for _ in 0..50 {
        match connect_async(url).await {
            Ok((stream, _resp)) => return stream,
            Err(err) => {
                last_err = err.to_string();
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
    panic!("failed to connect websocket {url}: {last_err}");
}

async fn recv_json(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .expect("timed out waiting for ws frame")
            .expect("ws stream closed")
            .expect("ws frame error");
        match frame {
            Message::Text(text) => {
                return serde_json::from_str(text.as_str()).expect("valid json text frame");
            }
            Message::Binary(bin) => {
                return serde_json::from_slice(&bin).expect("valid json binary frame");
            }
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Close(close) => panic!("unexpected ws close frame: {close:?}"),
        }
    }
}

async fn recv_response_by_id(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    id: &str,
) -> Value {
    for _ in 0..200 {
        let msg = recv_json(ws).await;
        if msg.get("id").and_then(Value::as_str) == Some(id) {
            return msg;
        }
    }
    panic!("timed out waiting for response id={id}");
}

fn scripted_runtime_options(
    work_dir: &TempDir,
    default_session_id: &str,
) -> WsSessionRuntimeOptions {
    WsSessionRuntimeOptions {
        work_dir: KaosPath::from(work_dir.path().to_path_buf()),
        default_session_id: default_session_id.to_string(),
        config: scripted_config(),
        model_name: None,
        thinking: Some(false),
        yolo: true,
        agent_file: None,
        mcp_configs: vec![],
        skills_dir: None,
        max_steps_per_turn: None,
        max_retries_per_step: None,
        max_ralph_iterations: None,
    }
}

#[tokio::test]
async fn test_wire_ws_initialize_and_prompt() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let work_dir = TempDir::new().expect("work dir");
    let _env = configure_scripted_env(&home_dir, &["text: hello from ws"]);
    let options = scripted_runtime_options(&work_dir, "default-session");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let listen_addr = listener.local_addr().expect("listener local addr");
    let server = WireWsServer::new(options, listen_addr, "/wire").expect("wire ws server");
    let server_task = tokio::spawn(async move { server.serve_with_listener(listener).await });

    let mut ws = connect_ws_with_retry(&format!("ws://{listen_addr}/wire")).await;
    ws.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": { "protocol_version": WIRE_PROTOCOL_VERSION }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send initialize");

    let init_resp = recv_json(&mut ws).await;
    assert_eq!(
        init_resp.get("id"),
        Some(&Value::String("init".to_string()))
    );
    assert_eq!(
        init_resp
            .get("result")
            .and_then(|v| v.get("protocol_version"))
            .and_then(Value::as_str),
        Some(WIRE_PROTOCOL_VERSION)
    );
    assert_eq!(
        init_resp
            .get("result")
            .and_then(|v| v.get("server"))
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str),
        Some(NAME)
    );
    assert_eq!(
        init_resp
            .get("result")
            .and_then(|v| v.get("server"))
            .and_then(|v| v.get("version"))
            .and_then(Value::as_str),
        Some(VERSION)
    );

    ws.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "prompt-1",
            "method": "prompt",
            "params": { "user_input": "hello" }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send prompt");

    let mut saw_event = false;
    let mut prompt_resp: Option<Value> = None;
    for _ in 0..200 {
        let msg = recv_json(&mut ws).await;
        if msg.get("method").and_then(Value::as_str) == Some("event") {
            saw_event = true;
        }
        if msg.get("id").and_then(Value::as_str) == Some("prompt-1") {
            prompt_resp = Some(msg);
            break;
        }
    }

    assert!(saw_event, "expected at least one event message");
    let prompt_resp = prompt_resp.expect("prompt response");
    assert_eq!(
        prompt_resp
            .get("result")
            .and_then(|v| v.get("status"))
            .and_then(Value::as_str),
        Some("finished")
    );

    ws.close(None).await.expect("close ws");
    server_task.abort();
    let join_err = server_task
        .await
        .expect_err("server task should be aborted for test shutdown");
    assert!(join_err.is_cancelled(), "unexpected join error: {join_err}");
}

#[tokio::test]
async fn test_wire_ws_external_tool_request_roundtrip() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let work_dir = TempDir::new().expect("work dir");
    let _env = configure_scripted_env(
        &home_dir,
        &[
            r#"tool_call: {"id":"tc-1","name":"open_in_ide","arguments":"{\"path\":\"/tmp/a\"}"}"#,
            "text: tool handled",
        ],
    );
    let options = scripted_runtime_options(&work_dir, "default-session");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let listen_addr = listener.local_addr().expect("listener local addr");
    let server = WireWsServer::new(options, listen_addr, "/wire").expect("wire ws server");
    let server_task = tokio::spawn(async move { server.serve_with_listener(listener).await });

    let mut ws = connect_ws_with_retry(&format!("ws://{listen_addr}/wire")).await;
    ws.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocol_version": WIRE_PROTOCOL_VERSION,
                "external_tools": [
                    {
                        "name": "open_in_ide",
                        "description": "Open file in IDE",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }
                    }
                ]
            }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send initialize");

    let init_resp = recv_json(&mut ws).await;
    assert_eq!(
        init_resp.get("id"),
        Some(&Value::String("init".to_string()))
    );
    assert_eq!(
        init_resp
            .get("result")
            .and_then(|v| v.get("external_tools"))
            .and_then(|v| v.get("accepted"))
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect::<Vec<_>>()),
        Some(vec!["open_in_ide"])
    );

    ws.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "prompt-1",
            "method": "prompt",
            "params": { "user_input": "use tool" }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send prompt");

    let mut saw_tool_request = false;
    let mut prompt_resp: Option<Value> = None;
    for _ in 0..200 {
        let msg = recv_json(&mut ws).await;
        if msg.get("method").and_then(Value::as_str) == Some("request") {
            let request_id = msg
                .get("id")
                .and_then(Value::as_str)
                .expect("request id")
                .to_string();
            let payload = msg
                .get("params")
                .and_then(|v| v.get("payload"))
                .expect("request payload");
            let tool_call_id = payload
                .get("id")
                .and_then(Value::as_str)
                .expect("tool call id")
                .to_string();
            saw_tool_request = true;

            ws.send(Message::Text(
                json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tool_call_id": tool_call_id,
                        "return_value": {
                            "is_error": false,
                            "output": "ok",
                            "message": "ok",
                            "display": []
                        }
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .expect("send tool result");
        }
        if msg.get("id").and_then(Value::as_str) == Some("prompt-1") {
            prompt_resp = Some(msg);
            break;
        }
    }

    assert!(saw_tool_request, "expected tool call request over wire ws");
    let prompt_resp = prompt_resp.expect("prompt response");
    assert_eq!(
        prompt_resp
            .get("result")
            .and_then(|v| v.get("status"))
            .and_then(Value::as_str),
        Some("finished")
    );

    ws.close(None).await.expect("close ws");
    server_task.abort();
    let join_err = server_task
        .await
        .expect_err("server task should be aborted for test shutdown");
    assert!(join_err.is_cancelled(), "unexpected join error: {join_err}");
}

#[tokio::test]
async fn test_wire_ws_same_session_rejects_second_client() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let work_dir = TempDir::new().expect("work dir");
    let _env = configure_scripted_env(&home_dir, &["text: hello"]);

    let options = scripted_runtime_options(&work_dir, "default-session");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let listen_addr = listener.local_addr().expect("listener local addr");
    let server = WireWsServer::new(options, listen_addr, "/wire").expect("wire ws multi server");
    let server_task = tokio::spawn(async move { server.serve_with_listener(listener).await });

    let mut first = connect_ws_with_retry(&format!("ws://{listen_addr}/wire?session_id=s1")).await;
    let second = connect_async(format!("ws://{listen_addr}/wire?session_id=s1")).await;
    match second {
        Err(WsError::Http(resp)) => {
            assert_eq!(
                resp.status(),
                tokio_tungstenite::tungstenite::http::StatusCode::CONFLICT
            );
        }
        other => panic!("expected HTTP 409 for second same-session client, got: {other:?}"),
    }

    first.close(None).await.expect("close first ws");
    server_task.abort();
    let join_err = server_task
        .await
        .expect_err("server task should be aborted for test shutdown");
    assert!(join_err.is_cancelled(), "unexpected join error: {join_err}");
}

#[tokio::test]
async fn test_wire_ws_allows_parallel_sessions() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let work_dir = TempDir::new().expect("work dir");
    let _env = configure_scripted_env(
        &home_dir,
        &[
            "text: session a response",
            "text: session b response",
            "text: fallback a",
            "text: fallback b",
        ],
    );

    let options = scripted_runtime_options(&work_dir, "default-session");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let listen_addr = listener.local_addr().expect("listener local addr");
    let server = WireWsServer::new(options, listen_addr, "/wire").expect("wire ws multi server");
    let server_task = tokio::spawn(async move { server.serve_with_listener(listener).await });

    let mut ws_a = connect_ws_with_retry(&format!("ws://{listen_addr}/wire?session_id=s-a")).await;
    let mut ws_b = connect_ws_with_retry(&format!("ws://{listen_addr}/wire?session_id=s-b")).await;

    ws_a.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "init-a",
            "method": "initialize",
            "params": { "protocol_version": WIRE_PROTOCOL_VERSION }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send initialize a");
    ws_b.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "init-b",
            "method": "initialize",
            "params": { "protocol_version": WIRE_PROTOCOL_VERSION }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send initialize b");

    let init_a = recv_response_by_id(&mut ws_a, "init-a").await;
    let init_b = recv_response_by_id(&mut ws_b, "init-b").await;
    assert_eq!(
        init_a
            .get("result")
            .and_then(|v| v.get("protocol_version"))
            .and_then(Value::as_str),
        Some(WIRE_PROTOCOL_VERSION)
    );
    assert_eq!(
        init_b
            .get("result")
            .and_then(|v| v.get("protocol_version"))
            .and_then(Value::as_str),
        Some(WIRE_PROTOCOL_VERSION)
    );

    ws_a.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "prompt-a",
            "method": "prompt",
            "params": { "user_input": "hello a" }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send prompt a");
    ws_b.send(Message::Text(
        json!({
            "jsonrpc": "2.0",
            "id": "prompt-b",
            "method": "prompt",
            "params": { "user_input": "hello b" }
        })
        .to_string()
        .into(),
    ))
    .await
    .expect("send prompt b");

    let resp_a = recv_response_by_id(&mut ws_a, "prompt-a").await;
    let resp_b = recv_response_by_id(&mut ws_b, "prompt-b").await;
    assert_eq!(
        resp_a
            .get("result")
            .and_then(|v| v.get("status"))
            .and_then(Value::as_str),
        Some("finished")
    );
    assert_eq!(
        resp_b
            .get("result")
            .and_then(|v| v.get("status"))
            .and_then(Value::as_str),
        Some("finished")
    );

    ws_a.close(None).await.expect("close ws a");
    ws_b.close(None).await.expect("close ws b");
    server_task.abort();
    let join_err = server_task
        .await
        .expect_err("server task should be aborted for test shutdown");
    assert!(join_err.is_cancelled(), "unexpected join error: {join_err}");
}

#[tokio::test]
async fn test_wire_ws_rejects_upgrade_when_runtime_init_fails_and_rolls_back_session() {
    let _lock = ENV_LOCK.lock().await;
    let home_dir = TempDir::new().expect("home dir");
    let work_dir = TempDir::new().expect("work dir");
    let _env = configure_scripted_env(&home_dir, &["text: hello"]);

    let mut options = scripted_runtime_options(&work_dir, "default-session");
    options.agent_file = Some(home_dir.path().join("missing-agent.yaml"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind listener");
    let listen_addr = listener.local_addr().expect("listener local addr");
    let server = WireWsServer::new(options, listen_addr, "/wire").expect("wire ws server");
    let server_task = tokio::spawn(async move { server.serve_with_listener(listener).await });

    let failed_session_id = "failed-session";
    let connect_result = connect_async(format!(
        "ws://{listen_addr}/wire?session_id={failed_session_id}"
    ))
    .await;
    match connect_result {
        Err(WsError::Http(resp)) => {
            assert_eq!(
                resp.status(),
                tokio_tungstenite::tungstenite::http::StatusCode::INTERNAL_SERVER_ERROR
            );
        }
        other => panic!("expected HTTP 500 when runtime init fails, got: {other:?}"),
    }

    let work_path = KaosPath::from(work_dir.path().to_path_buf());
    let session = Session::find(work_path, failed_session_id).await;
    assert!(
        session.is_none(),
        "expected failed websocket runtime init to roll back new session"
    );

    server_task.abort();
    let join_err = server_task
        .await
        .expect_err("server task should be aborted for test shutdown");
    assert!(join_err.is_cancelled(), "unexpected join error: {join_err}");
}
