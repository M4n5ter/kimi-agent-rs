use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use axum::extract::Query;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Router, http::StatusCode};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use kaos::KaosPath;
use kosong::chat_provider::ChatProviderError;
use kosong::tooling::tool_error;

use crate::app::{ConfigInput, CreateOptions, KimiCLI};
use crate::config::Config;
use crate::constant::{NAME, VERSION};
use crate::session::{Session, post_run as post_run_session};
use crate::session_id::normalize_session_id;
use crate::soul::kimisoul::KimiSoul;
use crate::soul::{LLMNotSet, LLMNotSupported, MaxStepsReached, RunCancelled, Soul, run_soul};
use crate::utils::{Queue, QueueShutDown};
use crate::wire::jsonrpc::{
    InitializeParams, JsonRpcErrorObject, JsonRpcErrorResponse, JsonRpcErrorResponseNullableId,
    JsonRpcMessage, JsonRpcSuccessResponse, PromptParams, build_event_message,
    build_request_message, error_codes, statuses,
};
use crate::wire::protocol::WIRE_PROTOCOL_VERSION;
use crate::wire::{
    ApprovalRequest, ApprovalResponse, ToolCallRequest, ToolResult, Wire, WireMessage,
};

const STDIO_BUFFER_LIMIT: usize = 100 * 1024 * 1024;

enum PendingRequest {
    Approval(ApprovalRequest),
    ToolCall(ToolCallRequest),
}

struct ActiveTurn {
    id: u64,
    cancel_token: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct WireRpcState {
    soul: Arc<KimiSoul>,
    write_queue: Queue<Value>,
    pending: Arc<tokio::sync::Mutex<HashMap<String, PendingRequest>>>,
    active_turn: Arc<tokio::sync::Mutex<Option<ActiveTurn>>>,
    next_turn_id: Arc<AtomicU64>,
}

impl WireRpcState {
    fn new(soul: Arc<KimiSoul>) -> Self {
        Self {
            soul,
            write_queue: Queue::new(),
            pending: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            active_turn: Arc::new(tokio::sync::Mutex::new(None)),
            next_turn_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn write_queue(&self) -> Queue<Value> {
        self.write_queue.clone()
    }

    fn session(&self) -> Session {
        self.soul.runtime().session.clone()
    }

    async fn handle_json_line(&self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }

        let msg_json: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => {
                error!("Invalid JSON line: {}", line);
                self.send_error_nullable(error_codes::PARSE_ERROR, "Invalid JSON format", None)
                    .await;
                return;
            }
        };

        self.handle_json_value(msg_json).await;
    }

    async fn handle_json_value(&self, msg_json: Value) {
        let response_hint = msg_json.get("method").is_none() && msg_json.get("id").is_some();
        let msg: JsonRpcMessage = match serde_json::from_value(msg_json.clone()) {
            Ok(msg) => msg,
            Err(err) => {
                if response_hint {
                    error!("Invalid JSON-RPC response: {:?}", err);
                } else {
                    error!("Invalid JSON-RPC message: {:?}", err);
                }
                let (code, message) = if response_hint {
                    (error_codes::INVALID_REQUEST, "Invalid response")
                } else {
                    (error_codes::INVALID_REQUEST, "Invalid request")
                };
                self.send_error_nullable(code, message, None).await;
                return;
            }
        };

        if let Some(version) = &msg.jsonrpc
            && version != "2.0"
        {
            self.send_error_nullable(error_codes::INVALID_REQUEST, "Invalid request", None)
                .await;
            return;
        }

        if msg.is_response() {
            if msg.result.is_none() && msg.error.is_none() {
                self.send_error_nullable(error_codes::INVALID_REQUEST, "Invalid response", None)
                    .await;
                return;
            }
            self.handle_response(&msg).await;
            return;
        }

        let method = match msg.method.as_deref() {
            Some(method) => method.to_string(),
            None => {
                error!("Invalid JSON-RPC inbound message: {:?}", msg);
                if let Some(id) = msg.id.clone() {
                    self.send_error(
                        id,
                        error_codes::METHOD_NOT_FOUND,
                        "Unexpected method received: None",
                    )
                    .await;
                }
                return;
            }
        };

        match method.as_str() {
            "initialize" => self.handle_initialize(msg).await,
            "prompt" => self.handle_prompt(msg).await,
            "cancel" => self.handle_cancel(msg).await,
            _ => {
                if let Some(id) = msg.id.clone() {
                    self.send_error(
                        id,
                        error_codes::METHOD_NOT_FOUND,
                        format!("Unexpected method received: {method}"),
                    )
                    .await;
                }
            }
        }
    }

    async fn handle_initialize(&self, msg: JsonRpcMessage) {
        let Some(id) = msg.id.clone() else {
            return;
        };
        if self.active_turn.lock().await.is_some() {
            self.send_error(
                id,
                error_codes::INVALID_STATE,
                "An agent turn is already in progress",
            )
            .await;
            return;
        }
        let params: InitializeParams = match msg
            .params
            .clone()
            .and_then(|params| serde_json::from_value(params).ok())
        {
            Some(params) => params,
            None => {
                self.send_error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid parameters for method `initialize`",
                )
                .await;
                return;
            }
        };

        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        if let Some(external_tools) = params.external_tools {
            let mut toolset = self.soul.agent().toolset.lock().await;
            for tool in external_tools {
                if toolset.has_builtin_tool(&tool.name) {
                    rejected
                        .push(json!({"name": tool.name, "reason": "conflicts with builtin tool"}));
                    continue;
                }
                match toolset.register_external_tool(&tool.name, &tool.description, tool.parameters)
                {
                    Ok(()) => accepted.push(tool.name),
                    Err(reason) => rejected.push(json!({"name": tool.name, "reason": reason})),
                }
            }
        }

        let slash_commands: Vec<Value> = self
            .soul
            .available_slash_commands()
            .into_iter()
            .map(|cmd| {
                json!({
                    "name": cmd.name,
                    "description": cmd.description,
                    "aliases": cmd.aliases,
                })
            })
            .collect();

        let mut result = json!({
            "protocol_version": WIRE_PROTOCOL_VERSION,
            "server": {"name": NAME, "version": VERSION},
            "slash_commands": slash_commands,
        });
        if !accepted.is_empty() || !rejected.is_empty() {
            result["external_tools"] = json!({
                "accepted": accepted,
                "rejected": rejected,
            });
        }

        let response = JsonRpcSuccessResponse {
            jsonrpc: "2.0",
            id,
            result,
        };
        let _ = self
            .write_queue
            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
    }

    async fn handle_prompt(&self, msg: JsonRpcMessage) {
        let Some(id) = msg.id.clone() else {
            return;
        };
        let params: PromptParams = match msg
            .params
            .clone()
            .and_then(|params| serde_json::from_value(params).ok())
        {
            Some(params) => params,
            None => {
                self.send_error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Invalid parameters for method `prompt`",
                )
                .await;
                return;
            }
        };

        let turn_id = self.next_turn_id.fetch_add(1, Ordering::Relaxed);
        let cancel_token = CancellationToken::new();
        let turn_cancel_token = cancel_token.clone();
        let active_turn_slot = Arc::clone(&self.active_turn);
        let soul = Arc::clone(&self.soul);
        let write_queue = self.write_queue.clone();
        let pending = Arc::clone(&self.pending);
        let wire_file = Some(self.soul.runtime().session.wire_file());
        let mut active_turn = self.active_turn.lock().await;
        if active_turn.is_some() {
            drop(active_turn);
            self.send_error(
                id,
                error_codes::INVALID_STATE,
                "An agent turn is already in progress",
            )
            .await;
            return;
        }

        let task = tokio::spawn(async move {
            let write_queue_for_stream = write_queue.clone();
            let pending_for_stream = Arc::clone(&pending);
            let run_handle = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(run_soul(
                    soul.as_ref(),
                    params.user_input,
                    move |wire| {
                        stream_wire_messages(
                            write_queue_for_stream.clone(),
                            Arc::clone(&pending_for_stream),
                            wire,
                        )
                    },
                    turn_cancel_token,
                    wire_file,
                ))
            });
            let run_result = match run_handle.await {
                Ok(result) => result,
                Err(err) => Err(anyhow::anyhow!("Wire run task failed: {err}")),
            };

            match run_result {
                Ok(()) => {
                    let response = JsonRpcSuccessResponse {
                        jsonrpc: "2.0",
                        id,
                        result: json!({"status": statuses::FINISHED}),
                    };
                    let _ = write_queue
                        .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                }
                Err(err) => {
                    if err.is::<LLMNotSet>() {
                        let response = JsonRpcErrorResponse {
                            jsonrpc: "2.0",
                            id,
                            error: JsonRpcErrorObject {
                                code: error_codes::LLM_NOT_SET,
                                message: "LLM is not set".to_string(),
                                data: None,
                            },
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    } else if err.is::<LLMNotSupported>() {
                        let response = JsonRpcErrorResponse {
                            jsonrpc: "2.0",
                            id,
                            error: JsonRpcErrorObject {
                                code: error_codes::LLM_NOT_SUPPORTED,
                                message: err.to_string(),
                                data: None,
                            },
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    } else if err.is::<ChatProviderError>() {
                        let response = JsonRpcErrorResponse {
                            jsonrpc: "2.0",
                            id,
                            error: JsonRpcErrorObject {
                                code: error_codes::CHAT_PROVIDER_ERROR,
                                message: err.to_string(),
                                data: None,
                            },
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    } else if let Some(MaxStepsReached { n_steps }) =
                        err.downcast_ref::<MaxStepsReached>()
                    {
                        let response = JsonRpcSuccessResponse {
                            jsonrpc: "2.0",
                            id,
                            result: json!({
                                "status": statuses::MAX_STEPS_REACHED,
                                "steps": n_steps,
                            }),
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    } else if err.is::<RunCancelled>() {
                        let response = JsonRpcSuccessResponse {
                            jsonrpc: "2.0",
                            id,
                            result: json!({"status": statuses::CANCELLED}),
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    } else {
                        let response = JsonRpcErrorResponse {
                            jsonrpc: "2.0",
                            id,
                            error: JsonRpcErrorObject {
                                code: error_codes::INTERNAL_ERROR,
                                message: err.to_string(),
                                data: None,
                            },
                        };
                        let _ = write_queue
                            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
                    }
                }
            }
            let mut slot = active_turn_slot.lock().await;
            if slot.as_ref().is_some_and(|turn| turn.id == turn_id) {
                *slot = None;
            }
        });
        *active_turn = Some(ActiveTurn {
            id: turn_id,
            cancel_token,
            task,
        });
    }

    async fn handle_cancel(&self, msg: JsonRpcMessage) {
        let Some(id) = msg.id.clone() else {
            return;
        };
        let cancel_token = self
            .active_turn
            .lock()
            .await
            .as_ref()
            .map(|turn| turn.cancel_token.clone());
        let Some(token) = cancel_token else {
            self.send_error(
                id,
                error_codes::INVALID_STATE,
                "No agent turn is in progress",
            )
            .await;
            return;
        };
        token.cancel();
        let response = JsonRpcSuccessResponse {
            jsonrpc: "2.0",
            id,
            result: json!({}),
        };
        let _ = self
            .write_queue
            .put_nowait(serde_json::to_value(response).unwrap_or(Value::Null));
    }

    async fn handle_response(&self, msg: &JsonRpcMessage) {
        let Some(id) = msg.id.clone() else {
            return;
        };
        let request = {
            let mut pending = self.pending.lock().await;
            pending.remove(&id)
        };
        let Some(request) = request else {
            error!("No pending request for response id={}", id);
            return;
        };

        match request {
            PendingRequest::Approval(req) => {
                if msg.error.is_some() {
                    req.resolve(crate::wire::ApprovalResponseKind::Reject);
                    return;
                }
                let result: ApprovalResponse = match msg
                    .result
                    .clone()
                    .and_then(|value| serde_json::from_value(value).ok())
                {
                    Some(result) => result,
                    None => {
                        error!(
                            "Invalid response result for request id={}: missing result",
                            id
                        );
                        req.resolve(crate::wire::ApprovalResponseKind::Reject);
                        return;
                    }
                };
                if result.request_id != req.id {
                    warn!(
                        "Approval response id mismatch: request={}, response={}",
                        req.id, result.request_id
                    );
                }
                req.resolve(result.response);
            }
            PendingRequest::ToolCall(req) => {
                if let Some(error) = &msg.error {
                    let return_value = tool_error("", error.message.clone(), "External tool error");
                    req.resolve(return_value);
                    return;
                }
                let tool_result: ToolResult = match msg
                    .result
                    .clone()
                    .and_then(|value| serde_json::from_value(value).ok())
                {
                    Some(result) => result,
                    None => {
                        error!("Invalid tool result for request id={}: missing result", id);
                        let return_value = tool_error(
                            "",
                            "Invalid tool result payload from client.",
                            "Invalid tool result",
                        );
                        req.resolve(return_value);
                        return;
                    }
                };
                if tool_result.tool_call_id != req.id {
                    warn!(
                        "Tool result id mismatch: request={}, result={}",
                        req.id, tool_result.tool_call_id
                    );
                }
                req.resolve(tool_result.return_value);
            }
        }
    }

    async fn send_error(&self, id: String, code: i64, message: impl Into<String>) {
        let response = JsonRpcErrorResponse {
            jsonrpc: "2.0",
            id,
            error: JsonRpcErrorObject {
                code,
                message: message.into(),
                data: None,
            },
        };
        if self
            .write_queue
            .put_nowait(serde_json::to_value(&response).unwrap_or(Value::Null))
            .is_err()
        {
            error!("Send queue shut down; dropping message: {:?}", response);
        }
    }

    async fn send_error_nullable(&self, code: i64, message: impl Into<String>, id: Option<String>) {
        let response = JsonRpcErrorResponseNullableId {
            jsonrpc: "2.0",
            id,
            error: JsonRpcErrorObject {
                code,
                message: message.into(),
                data: None,
            },
        };
        if self
            .write_queue
            .put_nowait(serde_json::to_value(&response).unwrap_or(Value::Null))
            .is_err()
        {
            error!("Send queue shut down; dropping message: {:?}", response);
        }
    }

    async fn reject_pending_requests(&self) {
        let pending = {
            let mut pending = self.pending.lock().await;
            std::mem::take(&mut *pending)
        };
        for (_, request) in pending {
            match request {
                PendingRequest::Approval(req) => {
                    req.resolve(crate::wire::ApprovalResponseKind::Reject);
                }
                PendingRequest::ToolCall(req) => {
                    let return_value = tool_error(
                        "",
                        "Wire connection closed before tool result was received.",
                        "Wire closed",
                    );
                    req.resolve(return_value);
                }
            }
        }
    }

    async fn shutdown(&self) {
        self.reject_pending_requests().await;

        cancel_and_wait_active_turn(&self.active_turn).await;

        self.reject_pending_requests().await;

        self.write_queue.shutdown(false);
    }
}

async fn cancel_and_wait_active_turn(
    active_turn_slot: &Arc<tokio::sync::Mutex<Option<ActiveTurn>>>,
) {
    let active_turn = {
        let mut active_turn = active_turn_slot.lock().await;
        active_turn.take()
    };
    if let Some(active_turn) = active_turn {
        active_turn.cancel_token.cancel();
        if let Err(err) = active_turn.task.await {
            warn!("Wire turn task join failed during shutdown: {:?}", err);
        }
    }
}

pub struct WireServer {
    rpc: WireRpcState,
}

pub type WireOverStdio = WireServer;

impl WireServer {
    pub fn new(soul: Arc<KimiSoul>) -> Self {
        Self {
            rpc: WireRpcState::new(soul),
        }
    }

    pub async fn serve(&self) -> anyhow::Result<()> {
        info!("Starting Wire server on stdio");
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = BufReader::with_capacity(STDIO_BUFFER_LIMIT, stdin);
        let mut writer = stdout;

        let write_queue = self.rpc.write_queue();
        let write_task = tokio::spawn(async move {
            loop {
                let msg = match write_queue.get().await {
                    Ok(msg) => msg,
                    Err(_) => {
                        debug!("Send queue shut down, stopping Wire server write loop");
                        break;
                    }
                };
                let line = match serde_json::to_string(&msg) {
                    Ok(line) => line,
                    Err(err) => {
                        error!("Wire server write loop error: {:?}", err);
                        continue;
                    }
                };
                if let Err(err) = writer.write_all(line.as_bytes()).await {
                    error!("Wire server write loop error: {:?}", err);
                    break;
                }
                if let Err(err) = writer.write_all(b"\n").await {
                    error!("Wire server write loop error: {:?}", err);
                    break;
                }
                let _ = writer.flush().await;
            }
        });

        let mut buf = Vec::new();
        loop {
            buf.clear();
            let n = reader.read_until(b'\n', &mut buf).await?;
            if n == 0 {
                info!("stdin closed, Wire server exiting");
                break;
            }
            let line = String::from_utf8_lossy(&buf);
            self.rpc.handle_json_line(&line).await;
        }

        self.rpc.shutdown().await;
        let _ = write_task.await;
        Ok(())
    }
}

pub struct WireWsServer {
    options: Arc<WsSessionRuntimeOptions>,
    listen_addr: SocketAddr,
    path: String,
}

#[derive(Clone)]
pub struct WsSessionRuntimeOptions {
    pub work_dir: KaosPath,
    pub default_session_id: String,
    pub config: Config,
    pub model_name: Option<String>,
    pub thinking: Option<bool>,
    pub yolo: bool,
    pub agent_file: Option<PathBuf>,
    pub mcp_configs: Vec<Value>,
    pub skills_dir: Option<KaosPath>,
    pub max_steps_per_turn: Option<i64>,
    pub max_retries_per_step: Option<i64>,
    pub max_ralph_iterations: Option<i64>,
}

impl WireWsServer {
    pub fn new(
        options: WsSessionRuntimeOptions,
        listen_addr: SocketAddr,
        path: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let path = path.into();
        let path = normalize_ws_path(&path)?;
        let mut options = options;
        options.default_session_id = normalize_session_id(&options.default_session_id)?;
        Ok(Self {
            options: Arc::new(options),
            listen_addr,
            path,
        })
    }

    pub async fn serve(&self) -> anyhow::Result<()> {
        let listener = tokio::net::TcpListener::bind(self.listen_addr).await?;
        self.serve_with_listener(listener).await
    }

    pub async fn serve_with_listener(
        &self,
        listener: tokio::net::TcpListener,
    ) -> anyhow::Result<()> {
        let bound_addr = listener.local_addr()?;
        info!(
            address = %bound_addr,
            path = %self.path,
            "Starting Wire server on websocket"
        );

        let state = Arc::new(WsServerState::new(Arc::clone(&self.options)));
        let app = Router::new()
            .route(&self.path, get(ws_upgrade_handler))
            .with_state(Arc::clone(&state));

        axum::serve(listener, app).await?;
        Ok(())
    }
}

struct WsServerState {
    options: Arc<WsSessionRuntimeOptions>,
    active_sessions: Mutex<HashSet<String>>,
}

impl WsServerState {
    fn new(options: Arc<WsSessionRuntimeOptions>) -> Self {
        Self {
            options,
            active_sessions: Mutex::new(HashSet::new()),
        }
    }

    fn resolve_session_id(&self, requested_session_id: Option<&str>) -> anyhow::Result<String> {
        match requested_session_id {
            Some(requested) => normalize_session_id(requested),
            None => Ok(self.options.default_session_id.clone()),
        }
    }

    fn lock_active_sessions(&self) -> MutexGuard<'_, HashSet<String>> {
        match self.active_sessions.lock() {
            Ok(guard) => guard,
            Err(err) => err.into_inner(),
        }
    }

    fn try_acquire_session(&self, session_id: &str) -> bool {
        let mut sessions = self.lock_active_sessions();
        if sessions.contains(session_id) {
            return false;
        }
        sessions.insert(session_id.to_string());
        true
    }

    fn release_session(&self, session_id: &str) {
        self.lock_active_sessions().remove(session_id);
    }

    async fn create_rpc(&self, session_id: &str) -> anyhow::Result<WireRpcState> {
        let options = &self.options;
        let found_session = Session::find(options.work_dir.clone(), session_id).await;
        let created_new_session = found_session.is_none();
        let session = match found_session {
            Some(session) => session,
            None => {
                Session::create(options.work_dir.clone(), Some(session_id.to_string()), None).await
            }
        };
        let rollback_session = if created_new_session {
            Some(session.clone())
        } else {
            None
        };

        let cli = KimiCLI::create(
            session,
            CreateOptions {
                config: Some(ConfigInput::Inline(Box::new(options.config.clone()))),
                model_name: options.model_name.clone(),
                thinking: options.thinking,
                yolo: options.yolo,
                agent_file: options.agent_file.clone(),
                mcp_configs: options.mcp_configs.clone(),
                skills_dir: options.skills_dir.clone(),
                max_steps_per_turn: options.max_steps_per_turn,
                max_retries_per_step: options.max_retries_per_step,
                max_ralph_iterations: options.max_ralph_iterations,
            },
        )
        .await;
        let cli = match cli {
            Ok(cli) => cli,
            Err(err) => {
                if let Some(rollback_session) = rollback_session
                    && let Err(post_run_err) = post_run_session(&rollback_session).await
                {
                    warn!(
                        session_id = %session_id,
                        "Failed to rollback newly created session after runtime init failure: {}",
                        post_run_err
                    );
                }
                return Err(err);
            }
        };

        Ok(WireRpcState::new(cli.soul()))
    }
}

async fn ws_upgrade_handler(
    State(state): State<Arc<WsServerState>>,
    Query(query): Query<HashMap<String, String>>,
    ws: WebSocketUpgrade,
) -> Response {
    let session_id = match state.resolve_session_id(query.get("session_id").map(String::as_str)) {
        Ok(session_id) => session_id,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };

    if !state.try_acquire_session(&session_id) {
        return (
            StatusCode::CONFLICT,
            format!("Session `{session_id}` already has an active websocket client."),
        )
            .into_response();
    }

    let prepared_rpc = match state.create_rpc(&session_id).await {
        Ok(rpc) => rpc,
        Err(err) => {
            error!(
                session_id = %session_id,
                "Failed to initialize wire websocket runtime: {}",
                err
            );
            state.release_session(&session_id);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to initialize websocket runtime: {err}"),
            )
                .into_response();
        }
    };

    let rpc_slot = Arc::new(tokio::sync::Mutex::new(Some(prepared_rpc)));
    let state_for_upgrade = Arc::clone(&state);
    let session_id_for_upgrade = session_id.clone();
    let rpc_slot_for_upgrade = Arc::clone(&rpc_slot);
    let state_for_failed_upgrade = Arc::clone(&state);
    let session_id_for_failed_upgrade = session_id.clone();
    let rpc_slot_for_failed_upgrade = Arc::clone(&rpc_slot);
    ws.on_failed_upgrade(move |err| {
        warn!(
            session_id = %session_id_for_failed_upgrade,
            "Wire websocket upgrade failed: {}",
            err
        );
        tokio::spawn(async move {
            if let Some(rpc) = rpc_slot_for_failed_upgrade.lock().await.take() {
                let session = rpc.session();
                rpc.shutdown().await;
                if let Err(post_run_err) = post_run_session(&session).await {
                    warn!(
                        session_id = %session_id_for_failed_upgrade,
                        "Failed to finalize session metadata after failed websocket upgrade: {}",
                        post_run_err
                    );
                }
            }
            state_for_failed_upgrade.release_session(&session_id_for_failed_upgrade);
        });
    })
    .on_upgrade(move |socket| async move {
        let rpc = rpc_slot_for_upgrade
            .lock()
            .await
            .take()
            .expect("prepared websocket runtime missing");
        handle_ws_socket(socket, state_for_upgrade, session_id_for_upgrade, rpc).await;
    })
    .into_response()
}

async fn handle_ws_socket(
    socket: WebSocket,
    state: Arc<WsServerState>,
    session_id: String,
    rpc: WireRpcState,
) {
    let (mut sender, mut receiver) = socket.split();

    let write_queue = rpc.write_queue();
    let write_task = tokio::spawn(async move {
        loop {
            let msg = match write_queue.get().await {
                Ok(msg) => msg,
                Err(_) => {
                    debug!("Send queue shut down, stopping websocket write loop");
                    break;
                }
            };

            let line = match serde_json::to_string(&msg) {
                Ok(line) => line,
                Err(err) => {
                    error!("Wire websocket write loop error: {:?}", err);
                    continue;
                }
            };

            if let Err(err) = sender.send(WsMessage::Text(line.into())).await {
                error!("Wire websocket write loop error: {:?}", err);
                break;
            }
        }
    });

    while let Some(frame) = receiver.next().await {
        match frame {
            Ok(WsMessage::Text(text)) => {
                rpc.handle_json_line(text.as_str()).await;
            }
            Ok(WsMessage::Binary(binary)) => match std::str::from_utf8(binary.as_ref()) {
                Ok(text) => rpc.handle_json_line(text).await,
                Err(_) => {
                    rpc.send_error_nullable(error_codes::INVALID_REQUEST, "Invalid request", None)
                        .await;
                }
            },
            Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {}
            Ok(WsMessage::Close(_)) => {
                info!(session_id = %session_id, "websocket closed by client");
                break;
            }
            Err(err) => {
                error!(
                    session_id = %session_id,
                    "Wire websocket read loop error: {:?}",
                    err
                );
                break;
            }
        }
    }

    let session = rpc.session();
    rpc.shutdown().await;
    if let Err(err) = post_run_session(&session).await {
        warn!(
            session_id = %session_id,
            "Failed to finalize session metadata after websocket close: {}",
            err
        );
    }
    state.release_session(&session_id);
    let _ = write_task.await;
}

fn normalize_ws_path(path: &str) -> anyhow::Result<String> {
    let normalized = path.trim();
    if normalized.is_empty() {
        anyhow::bail!("wire path cannot be empty");
    }
    if !normalized.starts_with('/') {
        anyhow::bail!("wire path must start with '/'");
    }
    if normalized.contains('?') || normalized.contains('#') {
        anyhow::bail!("wire path cannot contain query or fragment");
    }
    Ok(normalized.to_string())
}

async fn stream_wire_messages(
    write_queue: Queue<Value>,
    pending: Arc<tokio::sync::Mutex<HashMap<String, PendingRequest>>>,
    wire: Arc<Wire>,
) -> Result<(), QueueShutDown> {
    let ui_side = wire.ui_side(false);
    loop {
        let msg = ui_side.receive().await?;
        match msg {
            WireMessage::ApprovalRequest(request) => {
                request_approval(&write_queue, &pending, request).await;
            }
            WireMessage::ToolCallRequest(request) => {
                request_tool_call(&write_queue, &pending, request).await;
            }
            other => {
                let out = build_event_message(other);
                if write_queue
                    .put_nowait(serde_json::to_value(&out).unwrap_or(Value::Null))
                    .is_err()
                {
                    error!("Send queue shut down; dropping message: {:?}", out);
                }
            }
        }
    }
}

async fn request_approval(
    write_queue: &Queue<Value>,
    pending: &Arc<tokio::sync::Mutex<HashMap<String, PendingRequest>>>,
    request: ApprovalRequest,
) {
    let msg_id = request.id.clone();
    pending
        .lock()
        .await
        .insert(msg_id.clone(), PendingRequest::Approval(request.clone()));
    let out = build_request_message(msg_id, WireMessage::ApprovalRequest(request.clone()));
    let _ = write_queue.put_nowait(serde_json::to_value(out).unwrap_or(Value::Null));
    let _ = request.wait().await;
}

async fn request_tool_call(
    write_queue: &Queue<Value>,
    pending: &Arc<tokio::sync::Mutex<HashMap<String, PendingRequest>>>,
    request: ToolCallRequest,
) {
    let msg_id = request.id.clone();
    pending
        .lock()
        .await
        .insert(msg_id.clone(), PendingRequest::ToolCall(request.clone()));
    let out = build_request_message(msg_id, WireMessage::ToolCallRequest(request.clone()));
    let _ = write_queue.put_nowait(serde_json::to_value(out).unwrap_or(Value::Null));
    let _ = request.wait().await;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use tokio_util::sync::CancellationToken;

    use super::{ActiveTurn, cancel_and_wait_active_turn, normalize_ws_path};

    #[test]
    fn normalize_ws_path_accepts_valid_path() {
        assert_eq!(normalize_ws_path("/wire").unwrap(), "/wire");
        assert_eq!(normalize_ws_path(" /api/ws ").unwrap(), "/api/ws");
    }

    #[test]
    fn normalize_ws_path_rejects_invalid_path() {
        assert!(normalize_ws_path("").is_err());
        assert!(normalize_ws_path("wire").is_err());
        assert!(normalize_ws_path("/wire?x=1").is_err());
        assert!(normalize_ws_path("/wire#frag").is_err());
    }

    #[tokio::test]
    async fn cancel_and_wait_active_turn_waits_for_turn_exit() {
        let active_turn_slot = Arc::new(tokio::sync::Mutex::new(None));
        let cancel_token = CancellationToken::new();
        let task_cancel_token = cancel_token.clone();
        let task = tokio::spawn(async move {
            task_cancel_token.cancelled().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        *active_turn_slot.lock().await = Some(ActiveTurn {
            id: 1,
            cancel_token,
            task,
        });

        let started_at = Instant::now();
        cancel_and_wait_active_turn(&active_turn_slot).await;

        assert!(
            started_at.elapsed() >= Duration::from_millis(50),
            "cancel_and_wait_active_turn returned before turn task finished"
        );
        assert!(active_turn_slot.lock().await.is_none());
    }
}
