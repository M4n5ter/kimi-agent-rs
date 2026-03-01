use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use kaos::{ExecOptions, KaosPath};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, ClientInfo, Implementation,
};
use rmcp::service::{PeerRequestOptions, RunningService, ServiceError};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::auth::{AuthClient, AuthorizationManager};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, error, info, warn};

use kosong::message::ToolCall;
use kosong::tooling::error::{
    tool_not_found, tool_parse_error, tool_runtime_error, tool_validate_error,
};
use kosong::tooling::{CallableTool, Tool, ToolResult, ToolResultFuture, ToolReturnValue, Toolset};

use crate::constant::{NAME, VERSION};
use crate::exception::{InvalidToolError, MCPConfigError, MCPRuntimeError};
use crate::mcp::{get_mcp_credential_store, has_oauth_tokens};
use crate::mcp_http_proxy::KaosHttpProxyHandle;
use crate::mcp_transport::KaosChildProcessTransport;
use crate::soul::agent::Runtime;
use crate::soul::get_current_wire_or_none;
use crate::tools::utils::tool_rejected_error;
use crate::tools::{ToolDeps, load_tool};
use crate::wire::ToolCallRequest;
use kosong::tooling::mcp::convert_mcp_content;
use kosong::tooling::{tool_error, tool_ok};

tokio::task_local! {
    static CURRENT_TOOL_CALL: ToolCall;
}

pub fn with_current_tool_call<Fut>(
    tool_call: ToolCall,
    fut: Fut,
) -> impl Future<Output = Fut::Output>
where
    Fut: Future,
{
    CURRENT_TOOL_CALL.scope(tool_call, fut)
}

pub fn get_current_tool_call_or_none() -> Option<ToolCall> {
    CURRENT_TOOL_CALL.try_with(|call| call.clone()).ok()
}

pub struct KimiToolset {
    tools: HashMap<String, Arc<dyn CallableTool>>,
    external_tools: HashSet<String>,
    mcp_servers: HashMap<String, McpServerInfo>,
    mcp_loading_tasks: Vec<tokio::task::JoinHandle<Result<(), MCPRuntimeError>>>,
    mcp_tool_owners: HashMap<String, String>,
}

impl KimiToolset {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            external_tools: HashSet::new(),
            mcp_servers: HashMap::new(),
            mcp_loading_tasks: Vec::new(),
            mcp_tool_owners: HashMap::new(),
        }
    }

    pub fn add(&mut self, tool: Arc<dyn CallableTool>) {
        let base = tool.base();
        self.tools.insert(base.name.clone(), tool);
    }

    pub fn find(&self, name: &str) -> Option<Arc<dyn CallableTool>> {
        self.tools.get(name).cloned()
    }

    pub fn mcp_servers(&self) -> &HashMap<String, McpServerInfo> {
        &self.mcp_servers
    }

    pub fn has_builtin_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name) && !self.external_tools.contains(name)
    }

    pub fn register_external_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
    ) -> Result<(), String> {
        if self.tools.contains_key(name) && !self.external_tools.contains(name) {
            return Err("tool name conflicts with existing tool".to_string());
        }
        if let Err(err) = jsonschema::validator_for(&parameters) {
            return Err(err.to_string());
        }
        let tool = WireExternalTool::new(name, description, parameters);
        self.add(Arc::new(tool));
        self.external_tools.insert(name.to_string());
        Ok(())
    }

    fn register_mcp_tool(&mut self, server_name: &str, tool: McpTool) -> bool {
        let base = tool.base();
        let name = base.name.clone();
        let previous_owner = self.mcp_tool_owners.get(&name).cloned();

        if previous_owner.is_none() && self.tools.contains_key(&name) {
            warn!(
                "Skipping MCP tool '{}' from server '{}': name conflicts with non-MCP tool",
                name, server_name
            );
            return false;
        }

        if let Some(owner) = previous_owner
            && owner != server_name
        {
            warn!(
                "MCP tool '{}' from server '{}' overrides MCP tool from server '{}'",
                name, server_name, owner
            );
        }

        self.tools
            .insert(name.clone(), Arc::new(tool) as Arc<dyn CallableTool>);
        self.mcp_tool_owners.insert(name, server_name.to_string());
        true
    }

    pub fn load_tools(
        &mut self,
        tool_paths: &[String],
        runtime: &Runtime,
        toolset: Arc<tokio::sync::Mutex<KimiToolset>>,
    ) -> Result<(), InvalidToolError> {
        let deps = ToolDeps::new(runtime, toolset);
        let mut bad_tools = Vec::new();
        let mut good_tools = Vec::new();
        for tool_path in tool_paths {
            debug!("Loading tool: {}", tool_path);
            match load_tool(tool_path, &deps) {
                Ok(Some(tool)) => {
                    self.add(tool);
                    good_tools.push(tool_path.clone());
                }
                Ok(None) => {
                    info!("Skipping tool: {}", tool_path);
                }
                Err(_) => bad_tools.push(tool_path.clone()),
            }
        }
        info!("Loaded tools: {:?}", good_tools);
        if !bad_tools.is_empty() {
            return Err(InvalidToolError::new(format!(
                "Invalid tools: {:?}",
                bad_tools
            )));
        }
        Ok(())
    }

    pub async fn load_mcp_tools(
        &mut self,
        mcp_configs: &[serde_json::Value],
        runtime: &Runtime,
        toolset: Arc<tokio::sync::Mutex<KimiToolset>>,
    ) -> Result<(), anyhow::Error> {
        let mut servers_to_connect = Vec::new();
        for config in mcp_configs {
            let parsed = parse_mcp_config(config)
                .map_err(|err| anyhow::Error::new(MCPConfigError::new(err)))?;
            if parsed.is_empty() {
                debug!("Skipping empty MCP config: {:?}", config);
                continue;
            }
            for (name, server) in parsed {
                if let Some(existing) = self.mcp_servers.get(&name) {
                    if existing.config != server {
                        return Err(anyhow::Error::new(MCPConfigError::new(format!(
                            "Conflicting MCP config for server '{}'",
                            name
                        ))));
                    }

                    match existing.status {
                        McpServerStatus::Connected
                        | McpServerStatus::Connecting
                        | McpServerStatus::Pending => continue,
                        McpServerStatus::Failed | McpServerStatus::Unauthorized => {}
                    }
                }

                if let McpServerConfig::Http(http) = &server
                    && http.auth.as_deref() == Some("oauth")
                {
                    let authorized = has_oauth_tokens(&http.url)
                        .await
                        .map_err(|err| anyhow::anyhow!("Failed to read MCP auth tokens: {err}"))?;
                    if !authorized {
                        self.mcp_servers
                            .entry(name.clone())
                            .and_modify(|info| {
                                info.status = McpServerStatus::Unauthorized;
                                info.last_error = Some("OAuth authorization required".to_string());
                            })
                            .or_insert_with(|| {
                                let mut info = McpServerInfo::new(
                                    McpServerStatus::Unauthorized,
                                    server.clone(),
                                );
                                info.last_error = Some("OAuth authorization required".to_string());
                                info
                            });
                        warn!(
                            "Skipping OAuth MCP server '{}': not authorized. Run 'kimi-agent mcp auth {}' first.",
                            name, name
                        );
                        continue;
                    }
                }

                if backend_test_trace_enabled().await {
                    eprintln!("MCP config loaded server: {name}");
                }
                self.mcp_servers
                    .entry(name.clone())
                    .and_modify(|info| {
                        info.status = McpServerStatus::Pending;
                        info.last_error = None;
                    })
                    .or_insert_with(|| McpServerInfo::new(McpServerStatus::Pending, server));
                servers_to_connect.push(name);
            }
        }

        if servers_to_connect.is_empty() {
            return Ok(());
        }

        let toolset_ref = Arc::clone(&toolset);
        let runtime = runtime.clone();
        let task = tokio::spawn(async move {
            let mut failures: HashMap<String, String> = HashMap::new();
            for name in servers_to_connect {
                if backend_test_trace_enabled().await {
                    eprintln!("MCP connecting to server: {name}");
                }
                if let Err(err) = connect_mcp_server(&toolset_ref, &runtime, &name).await {
                    failures.insert(name.clone(), err.to_string());
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(MCPRuntimeError::new(format!(
                    "Failed to connect MCP servers: {failures:?}"
                )))
            }
        });
        self.mcp_loading_tasks.push(task);
        Ok(())
    }

    pub async fn wait_for_mcp_tools(&mut self) -> Result<(), anyhow::Error> {
        wait_mcp_loading_tasks(self.take_mcp_loading_tasks()).await
    }

    pub fn take_mcp_loading_tasks(
        &mut self,
    ) -> Vec<tokio::task::JoinHandle<Result<(), MCPRuntimeError>>> {
        std::mem::take(&mut self.mcp_loading_tasks)
    }

    pub async fn cleanup(&mut self) {
        let tasks = self.take_mcp_loading_tasks();
        for task in tasks {
            task.abort();
            let _ = task.await;
        }

        let mut servers = std::mem::take(&mut self.mcp_servers);
        for info in servers.values_mut() {
            if let Some(client) = info.client.take() {
                let mut service = client.service.lock().await;
                let _ = service.close().await;
                drop(service);
                if let Some(proxy) = client.proxy {
                    let mut proxy = proxy.lock().await;
                    let _ = proxy.close().await;
                }
            }
        }

        for name in self.mcp_tool_owners.keys().cloned().collect::<Vec<_>>() {
            self.tools.remove(&name);
        }
        self.mcp_tool_owners.clear();
    }
}

pub async fn wait_mcp_loading_tasks(
    tasks: Vec<tokio::task::JoinHandle<Result<(), MCPRuntimeError>>>,
) -> Result<(), anyhow::Error> {
    let mut failures = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => failures.push(err.to_string()),
            Err(err) => failures.push(err.to_string()),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Failed to connect MCP servers: {failures:?}"
        ))
    }
}

impl Default for KimiToolset {
    fn default() -> Self {
        Self::new()
    }
}

async fn backend_test_trace_enabled() -> bool {
    matches!(kaos::env_var("KIMI_TEST_TRACE").await, Ok(Some(value)) if value == "1")
}

impl Toolset for KimiToolset {
    fn tools(&self) -> Vec<Tool> {
        self.tools.values().map(|tool| tool.base()).collect()
    }

    fn handle(&self, tool_call: ToolCall) -> ToolResultFuture {
        let tool = match self.tools.get(&tool_call.function.name) {
            Some(tool) => Arc::clone(tool),
            None => {
                return ToolResultFuture::Immediate(ToolResult {
                    tool_call_id: tool_call.id,
                    return_value: tool_not_found(&tool_call.function.name),
                });
            }
        };

        let arguments = tool_call
            .function
            .arguments
            .clone()
            .unwrap_or_else(|| "{}".to_string());
        let args: Value = match serde_json::from_str(&arguments) {
            Ok(value) => value,
            Err(err) => {
                return ToolResultFuture::Immediate(ToolResult {
                    tool_call_id: tool_call.id,
                    return_value: tool_parse_error(&err.to_string()),
                });
            }
        };

        let tool_call_id = tool_call.id.clone();
        let schema = tool.base().parameters;
        let compiled = match jsonschema::validator_for(&schema) {
            Ok(compiled) => compiled,
            Err(err) => {
                return ToolResultFuture::Immediate(ToolResult {
                    tool_call_id,
                    return_value: tool_runtime_error(&err.to_string()),
                });
            }
        };
        if let Err(err) = compiled.validate(&args) {
            let msg = err.to_string();
            return ToolResultFuture::Immediate(ToolResult {
                tool_call_id,
                return_value: tool_validate_error(&msg),
            });
        }
        let tool_call_clone = tool_call.clone();
        let tool_ref = Arc::clone(&tool);
        let task = if crate::soul::get_current_wire_or_none().is_some() {
            crate::soul::spawn_with_current_wire(with_current_tool_call(
                tool_call_clone,
                async move {
                    let result = AssertUnwindSafe(tool_ref.call(args))
                        .catch_unwind()
                        .await
                        .unwrap_or_else(|panic| tool_runtime_error(&panic_message(panic)));
                    ToolResult {
                        tool_call_id,
                        return_value: result,
                    }
                },
            ))
        } else {
            tokio::task::spawn(with_current_tool_call(tool_call_clone, async move {
                let result = AssertUnwindSafe(tool_ref.call(args))
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|panic| tool_runtime_error(&panic_message(panic)));
                ToolResult {
                    tool_call_id,
                    return_value: result,
                }
            }))
        };
        ToolResultFuture::Pending(task)
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        message.to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "tool panicked".to_string()
    }
}

pub struct WireExternalTool {
    base: Tool,
}

impl WireExternalTool {
    pub fn new(name: &str, description: &str, parameters: Value) -> Self {
        let description = if description.trim().is_empty() {
            "No description provided."
        } else {
            description
        };
        Self {
            base: Tool::new(name, description, parameters),
        }
    }
}

#[async_trait::async_trait]
impl CallableTool for WireExternalTool {
    fn base(&self) -> Tool {
        self.base.clone()
    }

    async fn call(&self, _arguments: Value) -> ToolReturnValue {
        let tool_call = match get_current_tool_call_or_none() {
            Some(call) => call,
            None => {
                return ToolReturnValue {
                    is_error: true,
                    output: Default::default(),
                    message: "External tool calls must be invoked from a tool call context."
                        .to_string(),
                    display: Vec::new(),
                    extras: None,
                };
            }
        };

        let wire = match get_current_wire_or_none() {
            Some(wire) => wire,
            None => {
                error!(
                    "Wire is not available for external tool call: {}",
                    self.base.name
                );
                return ToolReturnValue {
                    is_error: true,
                    output: Default::default(),
                    message: "Wire is not available for external tool calls.".to_string(),
                    display: Vec::new(),
                    extras: None,
                };
            }
        };

        let request = ToolCallRequest::from_tool_call(&tool_call);
        wire.soul_side().send(request.clone().into());
        request.wait().await
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpServerStatus {
    Pending,
    Connecting,
    Connected,
    Failed,
    Unauthorized,
}

#[derive(Clone, Debug)]
struct McpClientHandle {
    peer: rmcp::Peer<RoleClient>,
    service: Arc<tokio::sync::Mutex<RunningService<RoleClient, ClientInfo>>>,
    proxy: Option<Arc<tokio::sync::Mutex<KaosHttpProxyHandle>>>,
}

#[derive(Clone, Debug)]
pub struct McpServerInfo {
    pub status: McpServerStatus,
    pub tools: Vec<String>,
    pub last_error: Option<String>,
    client: Option<McpClientHandle>,
    config: McpServerConfig,
}

impl McpServerInfo {
    fn new(status: McpServerStatus, config: McpServerConfig) -> Self {
        Self {
            status,
            config,
            tools: Vec::new(),
            last_error: None,
            client: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct McpConfig {
    #[serde(rename = "mcpServers", default)]
    mcp_servers: HashMap<String, McpServerConfig>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum McpServerConfig {
    Stdio(StdioServerConfig),
    Http(HttpServerConfig),
}

pub type McpToolSpec = rmcp::model::Tool;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpConnectionContext {
    pub cwd: KaosPath,
}

impl McpConnectionContext {
    pub fn from_runtime(runtime: &Runtime) -> Self {
        Self {
            cwd: runtime.session.work_dir.clone(),
        }
    }

    pub fn current() -> Self {
        Self {
            cwd: KaosPath::cwd(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct StdioServerConfig {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: Option<HashMap<String, String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct HttpServerConfig {
    pub url: String,
    #[serde(default)]
    pub transport: Option<String>,
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub auth: Option<String>,
}

pub fn parse_mcp_config(value: &Value) -> Result<HashMap<String, McpServerConfig>, String> {
    let config: McpConfig = serde_json::from_value(value.clone())
        .map_err(|err| format!("Invalid MCP config: {err}"))?;
    config
        .mcp_servers
        .into_iter()
        .map(|(name, server)| {
            canonicalize_mcp_server_config(server).map(|normalized| (name, normalized))
        })
        .collect()
}

fn build_client_info() -> ClientInfo {
    ClientInfo {
        client_info: Implementation {
            name: NAME.to_string(),
            title: None,
            version: VERSION.to_string(),
            description: None,
            icons: None,
            website_url: None,
        },
        ..Default::default()
    }
}

fn normalize_http_transport(transport: &Option<String>) -> Result<(), String> {
    canonical_http_transport(transport.as_deref()).map(|_| ())
}

fn canonical_http_transport(transport: Option<&str>) -> Result<&'static str, String> {
    match transport {
        None | Some("http") | Some("streamable-http") => Ok("streamable-http"),
        Some(other) => Err(format!("Unsupported transport: {other}")),
    }
}

fn canonicalize_mcp_server_config(server: McpServerConfig) -> Result<McpServerConfig, String> {
    match server {
        McpServerConfig::Stdio(mut stdio) => {
            if stdio.env.as_ref().is_some_and(HashMap::is_empty) {
                stdio.env = None;
            }
            Ok(McpServerConfig::Stdio(stdio))
        }
        McpServerConfig::Http(mut http) => {
            let transport = canonical_http_transport(http.transport.as_deref())?;
            http.transport = Some(transport.to_string());
            if http.headers.as_ref().is_some_and(HashMap::is_empty) {
                http.headers = None;
            }
            Ok(McpServerConfig::Http(http))
        }
    }
}

fn build_default_headers(headers: &Option<HashMap<String, String>>) -> Result<HeaderMap, String> {
    let mut map = HeaderMap::new();
    if let Some(custom_headers) = headers {
        for (key, value) in custom_headers {
            let header_name = HeaderName::from_bytes(key.as_bytes())
                .map_err(|err| format!("Invalid header name: {err}"))?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|err| format!("Invalid header value: {err}"))?;
            map.insert(header_name, header_value);
        }
    }
    Ok(map)
}

#[derive(Debug)]
enum McpClientError {
    Unauthorized(String),
    Other(String),
}

impl std::fmt::Display for McpClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpClientError::Unauthorized(message) => write!(f, "{message}"),
            McpClientError::Other(message) => write!(f, "{message}"),
        }
    }
}

async fn connect_mcp_client(
    config: &McpServerConfig,
    context: &McpConnectionContext,
) -> Result<McpClientHandle, McpClientError> {
    let (service, proxy) = match config {
        McpServerConfig::Stdio(server) => {
            let mut args = vec![server.command.clone()];
            args.extend(server.args.clone());
            let env_overrides: BTreeMap<String, String> =
                server.env.clone().unwrap_or_default().into_iter().collect();
            let process = kaos::exec_with_options(
                &args,
                ExecOptions {
                    cwd: Some(context.cwd.clone()),
                    env_overrides,
                },
            )
            .await
            .map_err(|err| McpClientError::Other(format!("Failed to spawn MCP server: {err}")))?;
            let transport = KaosChildProcessTransport::new(process).map_err(|err| {
                McpClientError::Other(format!("Failed to prepare MCP transport: {err}"))
            })?;
            let service = build_client_info().serve(transport).await.map_err(|err| {
                McpClientError::Other(format!("Failed to connect MCP server: {err}"))
            })?;
            (service, None)
        }
        McpServerConfig::Http(server) => {
            normalize_http_transport(&server.transport).map_err(McpClientError::Other)?;
            let headers = build_default_headers(&server.headers).map_err(McpClientError::Other)?;
            let (client, proxy) = build_http_client(headers).await?;

            if server.auth.as_deref() == Some("oauth") {
                let mut manager = AuthorizationManager::new(&server.url)
                    .await
                    .map_err(|err| McpClientError::Other(format!("OAuth init failed: {err}")))?;
                manager
                    .with_client(client.clone())
                    .map_err(|err| McpClientError::Other(format!("OAuth init failed: {err}")))?;
                manager.set_credential_store(get_mcp_credential_store(&server.url).await);
                let has_tokens = manager
                    .initialize_from_store()
                    .await
                    .map_err(|err| McpClientError::Other(format!("OAuth init failed: {err}")))?;
                if !has_tokens {
                    return Err(McpClientError::Unauthorized(
                        "OAuth authorization required".to_string(),
                    ));
                }
                let auth_client = AuthClient::new(client, manager);
                let transport = StreamableHttpClientTransport::with_client(
                    auth_client,
                    StreamableHttpClientTransportConfig::with_uri(server.url.clone()),
                );
                let service = build_client_info().serve(transport).await.map_err(|err| {
                    McpClientError::Other(format!("Failed to connect MCP server: {err}"))
                })?;
                (service, proxy)
            } else {
                let transport = StreamableHttpClientTransport::with_client(
                    client,
                    StreamableHttpClientTransportConfig::with_uri(server.url.clone()),
                );
                let service = build_client_info().serve(transport).await.map_err(|err| {
                    McpClientError::Other(format!("Failed to connect MCP server: {err}"))
                })?;
                (service, proxy)
            }
        }
    };

    let peer = service.peer().clone();
    Ok(McpClientHandle {
        peer,
        service: Arc::new(tokio::sync::Mutex::new(service)),
        proxy,
    })
}

async fn connect_mcp_server(
    toolset: &Arc<tokio::sync::Mutex<KimiToolset>>,
    runtime: &Runtime,
    server_name: &str,
) -> Result<(), MCPRuntimeError> {
    let config = {
        let mut guard = toolset.lock().await;
        let info = guard
            .mcp_servers
            .get_mut(server_name)
            .ok_or_else(|| MCPRuntimeError::new("MCP server not found"))?;
        if info.status != McpServerStatus::Pending {
            return Ok(());
        }
        info.status = McpServerStatus::Connecting;
        info.config.clone()
    };
    let connection_context = McpConnectionContext::from_runtime(runtime);

    let client = match connect_mcp_client(&config, &connection_context).await {
        Ok(client) => client,
        Err(McpClientError::Unauthorized(message)) => {
            let mut guard = toolset.lock().await;
            if let Some(info) = guard.mcp_servers.get_mut(server_name) {
                info.status = McpServerStatus::Unauthorized;
                info.last_error = Some(message);
            }
            warn!(
                "Skipping OAuth MCP server '{}': not authorized. Run 'kimi-agent mcp auth {}' first.",
                server_name, server_name
            );
            return Ok(());
        }
        Err(McpClientError::Other(err)) => {
            let mut guard = toolset.lock().await;
            if let Some(info) = guard.mcp_servers.get_mut(server_name) {
                info.status = McpServerStatus::Failed;
                info.last_error = Some(err.clone());
            }
            error!(
                "Failed to connect MCP server: {}, error: {}",
                server_name, err
            );
            if backend_test_trace_enabled().await {
                eprintln!("MCP connect error for {server_name}: {err}");
            }
            return Err(MCPRuntimeError::new(err));
        }
    };

    let tools = match client.peer.list_all_tools().await {
        Ok(tools) => tools,
        Err(err) => {
            let mut service = client.service.lock().await;
            let _ = service.close().await;
            let mut guard = toolset.lock().await;
            if let Some(info) = guard.mcp_servers.get_mut(server_name) {
                info.status = McpServerStatus::Failed;
                info.last_error = Some(err.to_string());
            }
            if backend_test_trace_enabled().await {
                eprintln!("MCP list tools error for {server_name}: {err}");
            }
            return Err(MCPRuntimeError::new(err.to_string()));
        }
    };
    if backend_test_trace_enabled().await {
        eprintln!("MCP server {server_name} listed {} tools", tools.len());
    }

    let mut guard = toolset.lock().await;
    let previous_tools = {
        let info = guard
            .mcp_servers
            .get_mut(server_name)
            .ok_or_else(|| MCPRuntimeError::new("MCP server not found"))?;
        info.status = McpServerStatus::Connected;
        info!("Connected MCP server: {}", server_name);
        info.last_error = None;
        info.client = Some(client.clone());
        std::mem::take(&mut info.tools)
    };

    let mut registered_tools = Vec::new();
    for tool in tools {
        let tool_name = tool.name.to_string();
        let wrapper = McpTool::new(server_name, tool, client.peer.clone(), runtime.clone());
        if guard.register_mcp_tool(server_name, wrapper) {
            registered_tools.push(tool_name);
        }
    }
    let registered_set: HashSet<_> = registered_tools.iter().cloned().collect();

    for old_tool in previous_tools {
        if registered_set.contains(&old_tool) {
            continue;
        }
        if guard.mcp_tool_owners.get(&old_tool).map(String::as_str) == Some(server_name) {
            guard.mcp_tool_owners.remove(&old_tool);
            guard.tools.remove(&old_tool);
        }
    }
    if let Some(info) = guard.mcp_servers.get_mut(server_name) {
        info.tools = registered_tools;
    }

    Ok(())
}

pub async fn list_mcp_tools(
    config: &McpServerConfig,
    context: &McpConnectionContext,
) -> Result<Vec<McpToolSpec>, String> {
    let client = connect_mcp_client(config, context)
        .await
        .map_err(|err| err.to_string())?;
    let tools = client
        .peer
        .list_all_tools()
        .await
        .map_err(|err| err.to_string())?;
    let mut service = client.service.lock().await;
    let _ = service.close().await;
    drop(service);
    if let Some(proxy) = client.proxy {
        let mut proxy = proxy.lock().await;
        let _ = proxy.close().await;
    }
    Ok(tools)
}

async fn build_http_client(
    headers: HeaderMap,
) -> Result<
    (
        reqwest::Client,
        Option<Arc<tokio::sync::Mutex<KaosHttpProxyHandle>>>,
    ),
    McpClientError,
> {
    let mut builder = reqwest::Client::builder().default_headers(headers);

    if kaos::get_current_kaos().name() == "local" {
        let client = builder
            .build()
            .map_err(|err| McpClientError::Other(format!("Failed to build HTTP client: {err}")))?;
        return Ok((client, None));
    }

    let proxy = KaosHttpProxyHandle::bind(kaos::get_current_kaos())
        .await
        .map_err(|err| McpClientError::Other(format!("Failed to start HTTP proxy: {err}")))?;
    let proxy_url = proxy.proxy_url();
    builder = builder
        .proxy(reqwest::Proxy::all(&proxy_url).map_err(|err| {
            McpClientError::Other(format!("Failed to configure HTTP proxy: {err}"))
        })?)
        .pool_max_idle_per_host(0);
    let client = builder
        .build()
        .map_err(|err| McpClientError::Other(format!("Failed to build HTTP client: {err}")))?;
    Ok((client, Some(Arc::new(tokio::sync::Mutex::new(proxy)))))
}

fn convert_mcp_tool_result(result: CallToolResult) -> ToolReturnValue {
    let mut content_parts = Vec::new();
    for part in result.content {
        let value = match serde_json::to_value(part) {
            Ok(value) => value,
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to parse MCP tool output: {err}"),
                    "MCP error",
                );
            }
        };
        match convert_mcp_content(&value) {
            Ok(part) => content_parts.push(part),
            Err(err) => {
                return tool_error(
                    "",
                    format!("Failed to parse MCP tool output: {err}"),
                    "MCP error",
                );
            }
        }
    }
    if result.is_error.unwrap_or(false) {
        tool_error(
            content_parts,
            "Tool returned an error. The output may be error message or incomplete output",
            "",
        )
    } else {
        tool_ok(content_parts, "", "")
    }
}

struct McpTool {
    base: Tool,
    peer: rmcp::Peer<RoleClient>,
    runtime: Runtime,
    action_name: String,
}

impl McpTool {
    fn new(
        server_name: &str,
        spec: McpToolSpec,
        peer: rmcp::Peer<RoleClient>,
        runtime: Runtime,
    ) -> Self {
        let description = format!(
            "This is an MCP (Model Context Protocol) tool from MCP server `{}`.\n\n{}",
            server_name,
            spec.description
                .as_ref()
                .map(|value| value.as_ref())
                .unwrap_or("No description provided.")
        );
        let input_schema = Value::Object(spec.input_schema.as_ref().clone());
        let base = Tool::new(spec.name.to_string(), &description, input_schema);
        Self {
            base,
            peer,
            runtime,
            action_name: format!("mcp:{}", spec.name),
        }
    }
}

#[async_trait::async_trait]
impl CallableTool for McpTool {
    fn base(&self) -> Tool {
        self.base.clone()
    }

    async fn call(&self, arguments: Value) -> ToolReturnValue {
        let description = format!("Call MCP tool `{}`.", self.base.name);
        let approved = self
            .runtime
            .approval
            .request(&self.base.name, &self.action_name, &description, None)
            .await
            .unwrap_or_default();
        if !approved {
            return tool_rejected_error();
        }

        let timeout_ms = self.runtime.config.mcp.client.tool_call_timeout_ms;
        let timeout_duration = Duration::from_millis(timeout_ms.max(1) as u64);

        let arguments = match arguments {
            Value::Null => None,
            Value::Object(map) => Some(map),
            _ => {
                return tool_parse_error("MCP tool arguments must be a JSON object");
            }
        };

        let request = CallToolRequest::new(CallToolRequestParams {
            meta: None,
            name: self.base.name.clone().into(),
            arguments,
            task: None,
        });
        let options = PeerRequestOptions {
            timeout: Some(timeout_duration),
            meta: None,
        };

        let response = match self
            .peer
            .send_request_with_option(request.into(), options)
            .await
        {
            Ok(handle) => handle.await_response().await,
            Err(err) => Err(err),
        };

        match response {
            Ok(rmcp::model::ServerResult::CallToolResult(result)) => {
                convert_mcp_tool_result(result)
            }
            Ok(other) => tool_error(
                "",
                format!("Unexpected MCP response: {other:?}"),
                "MCP error",
            ),
            Err(ServiceError::Timeout { .. }) => tool_error(
                "",
                format!(
                    concat!(
                        "Timeout while calling MCP tool `{}`. ",
                        "You may explain to the user that the timeout config is set too low."
                    ),
                    self.base.name
                ),
                "Timeout",
            ),
            Err(err) => tool_error("", err.to_string(), "MCP error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{McpServerConfig, parse_mcp_config};

    #[test]
    fn parse_mcp_config_canonicalizes_http_transport_aliases() {
        let config = json!({
            "mcpServers": {
                "a": { "url": "https://a.example/mcp" },
                "b": { "url": "https://b.example/mcp", "transport": "http" },
                "c": { "url": "https://c.example/mcp", "transport": "streamable-http" }
            }
        });

        let parsed = parse_mcp_config(&config).expect("parse should succeed");

        for name in ["a", "b", "c"] {
            let server = parsed.get(name).expect("server must exist");
            match server {
                McpServerConfig::Http(http) => {
                    assert_eq!(http.transport.as_deref(), Some("streamable-http"));
                }
                McpServerConfig::Stdio(_) => panic!("expected HTTP config"),
            }
        }
    }

    #[test]
    fn parse_mcp_config_canonicalizes_empty_maps() {
        let config = json!({
            "mcpServers": {
                "stdio": {
                    "command": "npx",
                    "args": [],
                    "env": {}
                },
                "http": {
                    "url": "https://example.com/mcp",
                    "headers": {}
                }
            }
        });

        let parsed = parse_mcp_config(&config).expect("parse should succeed");

        match parsed.get("stdio").expect("stdio must exist") {
            McpServerConfig::Stdio(stdio) => {
                assert_eq!(stdio.args, Vec::<String>::new());
                assert!(stdio.env.is_none());
            }
            McpServerConfig::Http(_) => panic!("expected stdio config"),
        }

        match parsed.get("http").expect("http must exist") {
            McpServerConfig::Http(http) => {
                assert!(http.headers.is_none());
                assert_eq!(http.transport.as_deref(), Some("streamable-http"));
            }
            McpServerConfig::Stdio(_) => panic!("expected HTTP config"),
        }
    }

    #[test]
    fn parse_mcp_config_rejects_unsupported_http_transport() {
        let config = json!({
            "mcpServers": {
                "bad": {
                    "url": "https://example.com/mcp",
                    "transport": "sse"
                }
            }
        });

        let err = parse_mcp_config(&config).expect_err("parse should fail");
        assert!(err.contains("Unsupported transport: sse"));
    }
}
