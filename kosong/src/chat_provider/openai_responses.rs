use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::chat_provider::{
    ChatProvider, ChatProviderError, ChatProviderErrorKind, StreamedMessage, ThinkingEffort,
    TokenUsage,
};
use crate::message::{
    ContentPart, Message, StreamedMessagePart, TextPart, ThinkPart, ToolCall, ToolCallFunction,
    ToolCallPart,
};
use crate::tooling::Tool;

type ByteStream = Pin<Box<dyn futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
type ParsedResponses = (Vec<StreamedMessagePart>, Option<String>, Option<TokenUsage>);

#[derive(Clone)]
pub struct OpenAIResponses {
    model: String,
    api_key: String,
    base_url: Url,
    stream: bool,
    client: Client,
    generation_kwargs: Map<String, Value>,
}

impl OpenAIResponses {
    pub fn new(
        model: impl Into<String>,
        api_key: Option<String>,
        base_url: Option<String>,
        default_headers: Option<HeaderMap>,
    ) -> Result<Self, ChatProviderError> {
        let api_key = api_key
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let mut base_url = base_url
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        if !base_url.ends_with('/') {
            base_url.push('/');
        }
        let base_url = Url::parse(&base_url).map_err(|err| {
            ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Invalid base URL: {err}"),
            )
        })?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static("KimiCLI"));
        if let Some(extra) = default_headers {
            for (k, v) in extra.iter() {
                if let Ok(value) = v.to_str() {
                    headers.insert(
                        k,
                        HeaderValue::from_str(value).unwrap_or_else(|_| v.clone()),
                    );
                } else {
                    headers.insert(k, v.clone());
                }
            }
        }

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        Ok(Self {
            model: model.into(),
            api_key,
            base_url,
            stream: true,
            client,
            generation_kwargs: Map::new(),
        })
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn with_generation_kwargs(mut self, kwargs: Map<String, Value>) -> Self {
        for (k, v) in kwargs {
            self.generation_kwargs.insert(k, v);
        }
        self
    }

    pub fn model_parameters(&self) -> Map<String, Value> {
        let mut params = Map::new();
        params.insert(
            "base_url".to_string(),
            Value::String(self.base_url.to_string()),
        );
        for (k, v) in &self.generation_kwargs {
            params.insert(k.clone(), v.clone());
        }
        params
    }

    fn use_developer_role(&self) -> bool {
        is_openai_model(&self.model)
    }
}

#[async_trait]
impl ChatProvider for OpenAIResponses {
    fn name(&self) -> &str {
        "openai-responses"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn thinking_effort(&self) -> Option<ThinkingEffort> {
        match self.generation_kwargs.get("reasoning_effort") {
            Some(Value::String(value)) => match value.as_str() {
                "low" | "minimal" => Some(ThinkingEffort::Low),
                "medium" => Some(ThinkingEffort::Medium),
                "high" => Some(ThinkingEffort::High),
                "xhigh" => Some(ThinkingEffort::XHigh),
                _ => Some(ThinkingEffort::Off),
            },
            Some(Value::Null) => Some(ThinkingEffort::Off),
            _ => None,
        }
    }

    async fn generate(
        &self,
        system_prompt: &str,
        tools: &[Tool],
        history: &[Message],
    ) -> Result<Box<dyn StreamedMessage>, ChatProviderError> {
        let mut input = Vec::new();
        let use_developer_role = self.use_developer_role();
        if !system_prompt.is_empty() {
            input.push(json!({
                "role": if use_developer_role {
                    "developer"
                } else {
                    "system"
                },
                "content": system_prompt,
            }));
        }
        for message in history {
            convert_history_message(message, &mut input, use_developer_role);
        }

        let mut tool_defs = Vec::new();
        for tool in tools {
            tool_defs.push(json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": false,
            }));
        }

        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("input".to_string(), Value::Array(input));
        body.insert("tools".to_string(), Value::Array(tool_defs));
        body.insert("stream".to_string(), Value::Bool(self.stream));
        body.insert("store".to_string(), Value::Bool(false));

        let mut generation_kwargs = self.generation_kwargs.clone();
        let reasoning_effort = generation_kwargs.remove("reasoning_effort");
        generation_kwargs.remove("reasoning");

        let mut include = vec![Value::String("reasoning.encrypted_content".to_string())];
        if let Some(Value::Array(values)) = generation_kwargs.remove("include") {
            for value in values {
                if value.as_str() == Some("reasoning.encrypted_content") {
                    continue;
                }
                include.push(value);
            }
        } else {
            generation_kwargs.remove("include");
        }
        for (k, v) in generation_kwargs {
            body.insert(k, v);
        }
        body.insert(
            "reasoning".to_string(),
            json!({
                "effort": reasoning_effort.unwrap_or(Value::Null),
                "summary": "auto",
            }),
        );
        body.insert("include".to_string(), Value::Array(include));

        let url = self
            .base_url
            .join("responses")
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        let mut request = self.client.post(url);
        if !self.api_key.is_empty() {
            request = request.header(AUTHORIZATION, format!("Bearer {}", self.api_key));
        }

        let resp = request
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Status(status.as_u16()),
                format!("OpenAI Responses API error ({status}): {text}"),
            ));
        }

        if self.stream {
            Ok(Box::new(OpenAIResponsesStreamedMessage::new_stream(resp)))
        } else {
            let value: Value = resp.json().await.map_err(map_reqwest_error)?;
            let (parts, message_id, usage) = parse_non_stream_response(&value);
            Ok(Box::new(OpenAIResponsesStreamedMessage::new_parts(
                parts, message_id, usage,
            )))
        }
    }

    fn with_thinking(&self, effort: ThinkingEffort) -> Box<dyn ChatProvider> {
        let mut kwargs = Map::new();
        let reasoning_effort = match effort {
            ThinkingEffort::Off => None,
            ThinkingEffort::Low => Some("low"),
            ThinkingEffort::Medium => Some("medium"),
            ThinkingEffort::High => Some("high"),
            ThinkingEffort::XHigh => Some("xhigh"),
        };
        if let Some(value) = reasoning_effort {
            kwargs.insert(
                "reasoning_effort".to_string(),
                Value::String(value.to_string()),
            );
        } else {
            kwargs.insert("reasoning_effort".to_string(), Value::Null);
        }

        let provider = self.clone().with_generation_kwargs(kwargs);
        Box::new(provider)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct OpenAIResponsesStreamedMessage {
    stream: Option<ByteStream>,
    buffer: String,
    parts: VecDeque<StreamedMessagePart>,
    id: Option<String>,
    usage: Option<TokenUsage>,
    saw_emittable_parts: bool,
    tool_call_by_item_id: HashMap<String, String>,
    tool_call_by_output_index: HashMap<i64, String>,
    emitted_tool_call_ids: HashSet<String>,
    tool_calls_with_argument_deltas: HashSet<String>,
    tool_calls_with_completion_arguments: HashSet<String>,
    tool_calls_with_inline_arguments: HashSet<String>,
    reasoning_delta_keys: HashSet<String>,
    reasoning_items_with_text_by_id: HashSet<String>,
    reasoning_items_with_text_by_output_index: HashSet<i64>,
}

impl OpenAIResponsesStreamedMessage {
    pub fn new_stream(resp: reqwest::Response) -> Self {
        let stream = resp.bytes_stream();
        Self {
            stream: Some(Box::pin(stream)),
            buffer: String::new(),
            parts: VecDeque::new(),
            id: None,
            usage: None,
            saw_emittable_parts: false,
            tool_call_by_item_id: HashMap::new(),
            tool_call_by_output_index: HashMap::new(),
            emitted_tool_call_ids: HashSet::new(),
            tool_calls_with_argument_deltas: HashSet::new(),
            tool_calls_with_completion_arguments: HashSet::new(),
            tool_calls_with_inline_arguments: HashSet::new(),
            reasoning_delta_keys: HashSet::new(),
            reasoning_items_with_text_by_id: HashSet::new(),
            reasoning_items_with_text_by_output_index: HashSet::new(),
        }
    }

    pub fn new_parts(
        parts: Vec<StreamedMessagePart>,
        id: Option<String>,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self {
            stream: None,
            buffer: String::new(),
            parts: parts.into(),
            id,
            usage,
            saw_emittable_parts: false,
            tool_call_by_item_id: HashMap::new(),
            tool_call_by_output_index: HashMap::new(),
            emitted_tool_call_ids: HashSet::new(),
            tool_calls_with_argument_deltas: HashSet::new(),
            tool_calls_with_completion_arguments: HashSet::new(),
            tool_calls_with_inline_arguments: HashSet::new(),
            reasoning_delta_keys: HashSet::new(),
            reasoning_items_with_text_by_id: HashSet::new(),
            reasoning_items_with_text_by_output_index: HashSet::new(),
        }
    }

    fn ingest_event(&mut self, value: &Value) {
        if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
            match event_type {
                "response.created" => {
                    if let Some(id) = value
                        .get("response")
                        .and_then(|response| response.get("id"))
                        .and_then(|id| id.as_str())
                    {
                        self.id = Some(id.to_string());
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str())
                        && !delta.is_empty()
                    {
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Text(
                                TextPart::new(delta),
                            )));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.refusal.delta" => {
                    if let Some(delta) = value
                        .get("delta")
                        .or_else(|| value.get("refusal"))
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Text(
                                TextPart::new(delta),
                            )));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.reasoning_summary_text.delta"
                | "response.reasoning_text.delta"
                | "response.reasoning.delta"
                | "response.reasoning_summary_text.done"
                | "response.reasoning_text.done"
                | "response.reasoning.done" => {
                    let is_done_event = event_type.ends_with(".done");
                    let text = value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .or_else(|| value.get("text").and_then(|v| v.as_str()));
                    if let Some(text) = text
                        && !text.is_empty()
                        && (!is_done_event || self.should_emit_reasoning_done_text(value))
                    {
                        if !is_done_event && let Some(key) = self.reasoning_event_key(value) {
                            self.reasoning_delta_keys.insert(key);
                        }
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Think(
                                ThinkPart {
                                    kind: "think".to_string(),
                                    think: text.to_string(),
                                    encrypted: None,
                                },
                            )));
                        self.mark_reasoning_item_text_emitted(value);
                        self.saw_emittable_parts = true;
                    }
                }
                "response.function_call_arguments.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str())
                        && !delta.is_empty()
                    {
                        let tool_call_id = self.resolve_tool_call_id(value);
                        if let Some(tool_call_id) = &tool_call_id {
                            self.tool_calls_with_argument_deltas
                                .insert(tool_call_id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                                arguments_part: Some(delta.to_string()),
                                tool_call_id,
                            }));
                    }
                }
                "response.function_call_arguments.done" => {
                    let tool_call_id = self.resolve_tool_call_id(value);
                    let arguments = value
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(ToString::to_string);
                    if let Some(arguments) = arguments
                        && self.should_emit_completion_arguments(tool_call_id.as_deref())
                    {
                        if let Some(tool_call_id) = &tool_call_id {
                            self.tool_calls_with_completion_arguments
                                .insert(tool_call_id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                                arguments_part: Some(arguments),
                                tool_call_id,
                            }));
                    }
                }
                "response.output_item.added" => {
                    if let Some(item) = value.get("item")
                        && let Some(tool_call) = parse_tool_call_item(item)
                    {
                        self.remember_tool_call_route(item, value, &tool_call.id);
                        self.emitted_tool_call_ids.insert(tool_call.id.clone());
                        if tool_call
                            .function
                            .arguments
                            .as_deref()
                            .is_some_and(|args| !args.is_empty())
                        {
                            self.tool_calls_with_inline_arguments
                                .insert(tool_call.id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCall(tool_call));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.output_item.done" => {
                    if let Some(item) = value.get("item") {
                        match item.get("type").and_then(|v| v.as_str()) {
                            Some("reasoning") => {
                                let item_id = item.get("id").and_then(|v| v.as_str());
                                let output_index =
                                    value.get("output_index").and_then(|v| v.as_i64());

                                let mut emitted_reasoning_text = false;
                                if !self.has_streamed_reasoning_text(item_id, output_index) {
                                    for think in parse_reasoning_parts_from_item(item) {
                                        self.parts.push_back(StreamedMessagePart::Content(
                                            ContentPart::Think(think),
                                        ));
                                        emitted_reasoning_text = true;
                                    }
                                }

                                if emitted_reasoning_text {
                                    if let Some(item_id) = item_id {
                                        self.reasoning_items_with_text_by_id
                                            .insert(item_id.to_string());
                                    }
                                    if let Some(output_index) = output_index {
                                        self.reasoning_items_with_text_by_output_index
                                            .insert(output_index);
                                    }
                                    self.saw_emittable_parts = true;
                                }

                                if !emitted_reasoning_text
                                    && let Some(encrypted) = item
                                        .get("encrypted_content")
                                        .and_then(|v| v.as_str())
                                        .filter(|v| !v.is_empty())
                                {
                                    self.parts.push_back(StreamedMessagePart::Content(
                                        ContentPart::Think(ThinkPart {
                                            kind: "think".to_string(),
                                            think: String::new(),
                                            encrypted: Some(encrypted.to_string()),
                                        }),
                                    ));
                                    self.saw_emittable_parts = true;
                                }
                            }
                            Some("function_call") => {
                                if let Some(tool_call) = parse_tool_call_item(item) {
                                    self.remember_tool_call_route(item, value, &tool_call.id);
                                    let tool_call_id = tool_call.id.clone();

                                    let already_emitted =
                                        self.emitted_tool_call_ids.contains(&tool_call_id);
                                    if !already_emitted {
                                        if tool_call
                                            .function
                                            .arguments
                                            .as_deref()
                                            .is_some_and(|args| !args.is_empty())
                                        {
                                            self.tool_calls_with_inline_arguments
                                                .insert(tool_call_id.clone());
                                        }
                                        self.parts.push_back(StreamedMessagePart::ToolCall(
                                            tool_call.clone(),
                                        ));
                                        self.emitted_tool_call_ids.insert(tool_call_id.clone());
                                        self.saw_emittable_parts = true;
                                    } else if let Some(arguments) =
                                        tool_call.function.arguments.filter(|args| !args.is_empty())
                                        && self
                                            .should_emit_completion_arguments(Some(&tool_call_id))
                                    {
                                        self.tool_calls_with_completion_arguments
                                            .insert(tool_call_id.clone());
                                        self.parts.push_back(StreamedMessagePart::ToolCallPart(
                                            ToolCallPart {
                                                arguments_part: Some(arguments),
                                                tool_call_id: Some(tool_call_id),
                                            },
                                        ));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "response.completed" => {
                    if let Some(response) = value.get("response") {
                        if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                            self.id = Some(id.to_string());
                        }
                        if let Some(usage) = response.get("usage") {
                            self.usage = parse_usage(usage);
                        }
                        if !self.saw_emittable_parts {
                            let mut final_parts = parse_parts_from_response(response);
                            if !final_parts.is_empty() {
                                self.saw_emittable_parts = true;
                            }
                            for part in final_parts.drain(..) {
                                self.parts.push_back(part);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(response) = value.get("response") {
            if let Some(usage) = response.get("usage") {
                self.usage = parse_usage(usage);
            }
        } else if let Some(usage) = value.get("usage") {
            self.usage = parse_usage(usage);
        }
    }

    fn remember_tool_call_route(&mut self, item: &Value, event: &Value, call_id: &str) {
        if let Some(item_id) = item.get("id").and_then(|v| v.as_str()) {
            self.tool_call_by_item_id
                .insert(item_id.to_string(), call_id.to_string());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            self.tool_call_by_output_index
                .insert(output_index, call_id.to_string());
        }
    }

    fn resolve_tool_call_id(&self, event: &Value) -> Option<String> {
        if let Some(call_id) = event.get("call_id").and_then(|v| v.as_str()) {
            return Some(call_id.to_string());
        }
        if let Some(call_id) = event
            .get("item")
            .and_then(|item| item.get("call_id"))
            .and_then(|v| v.as_str())
        {
            return Some(call_id.to_string());
        }
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
            && let Some(call_id) = self.tool_call_by_item_id.get(item_id)
        {
            return Some(call_id.clone());
        }
        if let Some(item_id) = event
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(|v| v.as_str())
            && let Some(call_id) = self.tool_call_by_item_id.get(item_id)
        {
            return Some(call_id.clone());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64())
            && let Some(call_id) = self.tool_call_by_output_index.get(&output_index)
        {
            return Some(call_id.clone());
        }
        None
    }

    fn should_emit_completion_arguments(&self, tool_call_id: Option<&str>) -> bool {
        let Some(tool_call_id) = tool_call_id else {
            return true;
        };
        !self.tool_calls_with_argument_deltas.contains(tool_call_id)
            && !self
                .tool_calls_with_completion_arguments
                .contains(tool_call_id)
            && !self.tool_calls_with_inline_arguments.contains(tool_call_id)
    }

    fn reasoning_event_key(&self, event: &Value) -> Option<String> {
        if let Some(summary_index) = event.get("summary_index").and_then(|v| v.as_i64()) {
            if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
                return Some(format!("item:{item_id}:summary:{summary_index}"));
            }
            if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
                return Some(format!("output:{output_index}:summary:{summary_index}"));
            }
            return Some(format!("summary:{summary_index}"));
        }

        let content_index = event.get("content_index").and_then(|v| v.as_i64())?;
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
            return Some(format!("item:{item_id}:content:{content_index}"));
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            return Some(format!("output:{output_index}:content:{content_index}"));
        }
        Some(format!("content:{content_index}"))
    }

    fn should_emit_reasoning_done_text(&self, event: &Value) -> bool {
        if let Some(key) = self.reasoning_event_key(event) {
            return !self.reasoning_delta_keys.contains(&key);
        }
        true
    }

    fn mark_reasoning_item_text_emitted(&mut self, event: &Value) {
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
            self.reasoning_items_with_text_by_id
                .insert(item_id.to_string());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            self.reasoning_items_with_text_by_output_index
                .insert(output_index);
        }
    }

    fn has_streamed_reasoning_text(
        &self,
        item_id: Option<&str>,
        output_index: Option<i64>,
    ) -> bool {
        item_id
            .map(|id| self.reasoning_items_with_text_by_id.contains(id))
            .unwrap_or(false)
            || output_index
                .map(|idx| {
                    self.reasoning_items_with_text_by_output_index
                        .contains(&idx)
                })
                .unwrap_or(false)
    }
}

#[async_trait]
impl StreamedMessage for OpenAIResponsesStreamedMessage {
    async fn next_part(&mut self) -> Result<Option<StreamedMessagePart>, ChatProviderError> {
        loop {
            if let Some(part) = self.parts.pop_front() {
                return Ok(Some(part));
            }
            let stream = match &mut self.stream {
                Some(stream) => stream,
                None => return Ok(None),
            };
            match stream.next().await {
                Some(Ok(bytes)) => {
                    let chunk = String::from_utf8_lossy(&bytes);
                    self.buffer.push_str(&chunk);
                    while let Some(pos) = self.buffer.find('\n') {
                        let line = self.buffer[..pos].trim().to_string();
                        self.buffer = self.buffer[pos + 1..].to_string();
                        if line.is_empty() {
                            continue;
                        }
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                self.stream = None;
                                break;
                            }
                            let value: Value = serde_json::from_str(data).map_err(|err| {
                                ChatProviderError::new(
                                    ChatProviderErrorKind::Other,
                                    err.to_string(),
                                )
                            })?;
                            self.ingest_event(&value);
                        }
                    }
                }
                Some(Err(err)) => return Err(map_reqwest_error(err)),
                None => {
                    self.stream = None;
                    return Ok(None);
                }
            }
        }
    }

    fn id(&self) -> Option<String> {
        self.id.clone()
    }

    fn usage(&self) -> Option<TokenUsage> {
        self.usage.clone()
    }
}

fn convert_history_message(message: &Message, input: &mut Vec<Value>, use_developer_role: bool) {
    if matches!(message.role, crate::message::Role::Tool) {
        let output = convert_tool_message_output(&message.content);
        input.push(json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id.clone().unwrap_or_default(),
            "output": output,
        }));
        return;
    }

    let role = match message.role {
        crate::message::Role::System => {
            if use_developer_role {
                "developer"
            } else {
                "system"
            }
        }
        crate::message::Role::User => "user",
        crate::message::Role::Assistant => "assistant",
        crate::message::Role::Tool => "tool",
    };

    convert_non_tool_message_content(role, &message.content, input);

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            input.push(json!({
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments.clone().unwrap_or_else(|| "{}".to_string()),
            }));
        }
    }
}

fn convert_non_tool_message_content(role: &str, content: &[ContentPart], input: &mut Vec<Value>) {
    if content.is_empty() {
        return;
    }

    let mut pending_parts = Vec::new();
    let mut index = 0usize;
    while index < content.len() {
        match &content[index] {
            ContentPart::Think(think) => {
                flush_pending_message_parts(role, &mut pending_parts, input);

                let encrypted = think.encrypted.clone();
                let mut summary = vec![json!({
                    "type": "summary_text",
                    "text": think.think.clone(),
                })];
                index += 1;
                while index < content.len() {
                    match &content[index] {
                        ContentPart::Think(next) if next.encrypted == encrypted => {
                            summary.push(json!({
                                "type": "summary_text",
                                "text": next.think.clone(),
                            }));
                            index += 1;
                        }
                        _ => break,
                    }
                }

                let mut reasoning_item = Map::new();
                reasoning_item.insert("type".to_string(), Value::String("reasoning".to_string()));
                reasoning_item.insert("summary".to_string(), Value::Array(summary));
                if let Some(encrypted_content) = encrypted {
                    reasoning_item.insert(
                        "encrypted_content".to_string(),
                        Value::String(encrypted_content),
                    );
                }
                input.push(Value::Object(reasoning_item));
            }
            other => {
                pending_parts.push(other.clone());
                index += 1;
            }
        }
    }

    flush_pending_message_parts(role, &mut pending_parts, input);
}

fn flush_pending_message_parts(
    role: &str,
    pending_parts: &mut Vec<ContentPart>,
    input: &mut Vec<Value>,
) {
    if pending_parts.is_empty() {
        return;
    }

    let content = if role == "assistant" {
        convert_content_parts_to_output_items(pending_parts)
    } else {
        convert_content_parts_to_input_items(pending_parts)
    };
    pending_parts.clear();

    if content.is_empty() {
        return;
    }

    input.push(json!({
        "type": "message",
        "role": role,
        "content": content,
    }));
}

fn convert_tool_message_output(parts: &[ContentPart]) -> Value {
    let output_items = convert_content_parts_to_function_output_items(parts);
    if output_items.is_empty() {
        Value::String(extract_text_from_parts(parts, "\n"))
    } else {
        Value::Array(output_items)
    }
}

fn convert_content_parts_to_input_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) if !text.text.is_empty() => {
                converted.push(json!({
                    "type": "input_text",
                    "text": text.text.clone(),
                }));
            }
            ContentPart::ImageUrl(image) => {
                converted.push(json!({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": image.image_url.url.clone(),
                }));
            }
            ContentPart::AudioUrl(audio) => {
                if let Some(mapped) = map_audio_url_to_input_item(&audio.audio_url.url) {
                    converted.push(mapped);
                }
            }
            ContentPart::Text(_) => {}
            ContentPart::Think(_) | ContentPart::VideoUrl(_) => {}
        }
    }
    converted
}

fn convert_content_parts_to_output_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        if let ContentPart::Text(text) = part
            && !text.text.is_empty()
        {
            converted.push(json!({
                "type": "output_text",
                "text": text.text.clone(),
                "annotations": [],
            }));
        }
    }
    converted
}

fn convert_content_parts_to_function_output_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) if !text.text.is_empty() => {
                converted.push(json!({
                    "type": "input_text",
                    "text": text.text.clone(),
                }));
            }
            ContentPart::ImageUrl(image) => {
                converted.push(json!({
                    "type": "input_image",
                    "image_url": image.image_url.url.clone(),
                }));
            }
            ContentPart::AudioUrl(audio) => {
                if let Some(mapped) = map_audio_url_to_file_content_item(&audio.audio_url.url) {
                    converted.push(mapped);
                }
            }
            ContentPart::Text(_) => {}
            ContentPart::Think(_) | ContentPart::VideoUrl(_) => {}
        }
    }
    converted
}

fn map_audio_url_to_input_item(url: &str) -> Option<Value> {
    if let Some((media_type, data)) = parse_data_url(url) {
        if !media_type.starts_with("audio/") {
            return None;
        }
        let subtype = media_type
            .split('/')
            .nth(1)?
            .split(';')
            .next()?
            .to_ascii_lowercase();
        let ext = match subtype.as_str() {
            "mp3" | "mpeg" => "mp3",
            "wav" => "wav",
            _ => return None,
        };
        return Some(json!({
            "type": "input_file",
            "file_data": data,
            "filename": format!("inline.{ext}"),
        }));
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "input_file",
            "file_url": url,
        }));
    }

    None
}

fn map_audio_url_to_file_content_item(url: &str) -> Option<Value> {
    if let Some((media_type, data)) = parse_data_url(url) {
        if !media_type.starts_with("audio/") {
            return None;
        }
        return Some(json!({
            "type": "input_file",
            "file_data": data,
        }));
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "input_file",
            "file_url": url,
        }));
    }

    None
}

fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let data = url.strip_prefix("data:")?;
    data.split_once(',')
}

fn is_openai_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    if model.is_empty() || model.contains('/') {
        return false;
    }

    model.starts_with("gpt-")
        || model.starts_with("chatgpt-")
        || model.starts_with("codex-")
        || model == "o1"
        || model.starts_with("o1-")
        || model == "o3"
        || model.starts_with("o3-")
        || model == "o4"
        || model.starts_with("o4-")
        || model == "o5"
        || model.starts_with("o5-")
}

fn extract_text_from_parts(parts: &[ContentPart], sep: &str) -> String {
    let mut text_parts = Vec::new();
    for part in parts {
        if let ContentPart::Text(text) = part {
            text_parts.push(text.text.clone());
        }
    }
    text_parts.join(sep)
}

fn parse_non_stream_response(value: &Value) -> ParsedResponses {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let usage = value.get("usage").and_then(parse_usage);
    let parts = parse_parts_from_response(value);
    (parts, id, usage)
}

fn parse_parts_from_response(response: &Value) -> Vec<StreamedMessagePart> {
    let mut parts = Vec::new();

    if let Some(output_text) = response
        .get("output_text")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
    {
        parts.push(StreamedMessagePart::Content(ContentPart::Text(
            TextPart::new(output_text),
        )));
    }

    if let Some(output) = response.get("output").and_then(|v| v.as_array()) {
        for item in output {
            let item_type = item
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            match item_type {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for block in content {
                            let block_type = block
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();
                            if matches!(block_type, "output_text" | "text" | "refusal")
                                && let Some(text) = block
                                    .get("text")
                                    .or_else(|| block.get("refusal"))
                                    .and_then(|v| v.as_str())
                                    .filter(|v| !v.is_empty())
                            {
                                parts.push(StreamedMessagePart::Content(ContentPart::Text(
                                    TextPart::new(text),
                                )));
                            }
                        }
                    }
                }
                "reasoning" => {
                    for think in parse_reasoning_parts_from_item(item) {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(think)));
                    }
                }
                "function_call" => {
                    if let Some(tool_call) = parse_tool_call_item(item) {
                        parts.push(StreamedMessagePart::ToolCall(tool_call));
                    }
                }
                _ => {}
            }
        }
    }

    parts
}

fn parse_reasoning_parts_from_item(item: &Value) -> Vec<ThinkPart> {
    let encrypted = item
        .get("encrypted_content")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string());

    let content_texts = item
        .get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| parse_reasoning_text_blocks(blocks, true))
        .unwrap_or_default();
    let summary_texts = item
        .get("summary")
        .and_then(|v| v.as_array())
        .map(|blocks| parse_reasoning_text_blocks(blocks, false))
        .unwrap_or_default();

    let texts = if !content_texts.is_empty() {
        content_texts
    } else {
        summary_texts
    };

    texts
        .into_iter()
        .map(|text| ThinkPart {
            kind: "think".to_string(),
            think: text,
            encrypted: encrypted.clone(),
        })
        .collect()
}

fn parse_reasoning_text_blocks(blocks: &[Value], prefer_content_types: bool) -> Vec<String> {
    let mut texts = Vec::new();
    for block in blocks {
        let block_type = block
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let allowed = if prefer_content_types {
            matches!(block_type, "" | "reasoning_text" | "text")
        } else {
            matches!(block_type, "" | "summary_text" | "text")
        };
        if !allowed {
            continue;
        }
        if let Some(text) = block
            .get("text")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
        {
            texts.push(text.to_string());
        }
    }
    texts
}

fn parse_tool_call_item(item: &Value) -> Option<ToolCall> {
    let item_type = item.get("type").and_then(|v| v.as_str())?;
    if item_type != "function_call" {
        return None;
    }

    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let name = item
        .get("name")
        .or_else(|| item.get("function").and_then(|v| v.get("name")))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())?;

    let arguments = item
        .get("arguments")
        .or_else(|| item.get("function").and_then(|v| v.get("arguments")))
        .and_then(stringify_json_value);

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction { name, arguments },
        extras: None,
    })
}

fn stringify_json_value(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if value.is_null() {
        return None;
    }
    serde_json::to_string(value).ok()
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let input_tokens = value
        .get("input_tokens")
        .and_then(|v| v.as_i64())
        .or_else(|| value.get("prompt_tokens").and_then(|v| v.as_i64()));
    let output_tokens = value
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .or_else(|| value.get("completion_tokens").and_then(|v| v.as_i64()));
    if input_tokens.is_none() && output_tokens.is_none() {
        return None;
    }

    let input_tokens = input_tokens.unwrap_or(0);
    let output_tokens = output_tokens.unwrap_or(0);

    let details = value.get("input_tokens_details");
    let cache_read = details
        .and_then(|v| v.get("cached_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let cache_creation = details
        .and_then(|v| v.get("cache_creation_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let input_other = (input_tokens - cache_read - cache_creation).max(0);

    Some(TokenUsage {
        input_other,
        output: output_tokens,
        input_cache_read: cache_read,
        input_cache_creation: cache_creation,
    })
}

fn map_reqwest_error(err: reqwest::Error) -> ChatProviderError {
    if err.is_timeout() {
        ChatProviderError::new(ChatProviderErrorKind::Timeout, err.to_string())
    } else if err.is_connect() {
        ChatProviderError::new(ChatProviderErrorKind::Connection, err.to_string())
    } else {
        ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_non_stream_response_extracts_text_usage_and_tool_call() {
        let value = json!({
            "id": "resp-1",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 5,
                    "cache_creation_tokens": 3
                }
            },
            "output": [
                {
                    "type": "message",
                    "content": [
                        { "type": "output_text", "text": "hello" }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "sum",
                    "arguments": "{\"a\":1}"
                }
            ]
        });

        let (parts, id, usage) = parse_non_stream_response(&value);
        assert_eq!(id.as_deref(), Some("resp-1"));
        let usage = usage.expect("usage");
        assert_eq!(usage.input_other, 12);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_parse_usage_returns_none_for_null_or_empty_payload() {
        assert_eq!(parse_usage(&Value::Null), None);
        assert_eq!(parse_usage(&json!({})), None);
    }

    #[test]
    fn test_is_openai_model_for_role_mapping() {
        assert!(is_openai_model("gpt-5-codex"));
        assert!(is_openai_model("o3"));
        assert!(is_openai_model("chatgpt-4o-latest"));
        assert!(!is_openai_model("claude-sonnet-4"));
        assert!(!is_openai_model("openai/gpt-5"));
    }

    #[test]
    fn test_parse_non_stream_response_prefers_reasoning_content() {
        let value = json!({
            "output": [
                {
                    "type": "reasoning",
                    "encrypted_content": "enc-1",
                    "content": [
                        { "type": "reasoning_text", "text": "full-trace" }
                    ],
                    "summary": [
                        { "type": "summary_text", "text": "summary-trace" }
                    ]
                }
            ]
        });

        let (parts, _, _) = parse_non_stream_response(&value);
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts,
            vec![StreamedMessagePart::Content(ContentPart::Think(
                ThinkPart {
                    kind: "think".to_string(),
                    think: "full-trace".to_string(),
                    encrypted: Some("enc-1".to_string()),
                }
            ))]
        );
    }

    #[test]
    fn test_with_thinking_supports_xhigh_effort() {
        let provider = OpenAIResponses::new("gpt-5", Some("k".to_string()), None, None)
            .expect("provider")
            .with_generation_kwargs({
                let mut kwargs = Map::new();
                kwargs.insert(
                    "reasoning_effort".to_string(),
                    Value::String("xhigh".to_string()),
                );
                kwargs
            });
        assert_eq!(provider.thinking_effort(), Some(ThinkingEffort::XHigh));

        let xhigh = provider.with_thinking(ThinkingEffort::XHigh);
        let typed = xhigh
            .as_any()
            .downcast_ref::<OpenAIResponses>()
            .expect("openai responses");
        assert_eq!(
            typed.generation_kwargs.get("reasoning_effort"),
            Some(&Value::String("xhigh".to_string()))
        );
    }

    #[test]
    fn test_reasoning_summary_done_is_deduplicated_after_delta() {
        let mut stream = OpenAIResponsesStreamedMessage::new_parts(Vec::new(), None, None);

        stream.ingest_event(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "summary"
        }));
        stream.ingest_event(&json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "text": "summary"
        }));

        assert_eq!(stream.parts.len(), 1);
    }
}
