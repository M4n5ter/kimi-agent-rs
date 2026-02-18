use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};

use crate::chat_provider::{
    ChatProvider, ChatProviderError, ChatProviderErrorKind, StreamedMessage, ThinkingEffort,
    TokenUsage,
};
use crate::message::{
    ContentPart, Message, Role, StreamedMessagePart, TextPart, ThinkPart, ToolCall,
    ToolCallFunction, ToolCallPart,
};
use crate::tooling::Tool;

type ByteStream = Pin<Box<dyn futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
type ParsedAnthropicResponse = (Vec<StreamedMessagePart>, Option<String>, Option<TokenUsage>);
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const DEFAULT_MAX_TOKENS: i64 = 50_000;

#[derive(Clone)]
pub struct Anthropic {
    model: String,
    api_key: String,
    base_url: Url,
    stream: bool,
    client: Client,
    generation_kwargs: Map<String, Value>,
}

impl Anthropic {
    pub fn new(
        model: impl Into<String>,
        api_key: Option<String>,
        base_url: Option<String>,
        default_headers: Option<HeaderMap>,
    ) -> Result<Self, ChatProviderError> {
        let api_key = api_key
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
            .unwrap_or_default();
        let mut base_url = base_url
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("ANTHROPIC_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
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
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
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
            generation_kwargs: {
                let mut kwargs = Map::new();
                kwargs.insert("max_tokens".to_string(), Value::from(DEFAULT_MAX_TOKENS));
                kwargs.insert(
                    "beta_features".to_string(),
                    Value::Array(vec![Value::String(INTERLEAVED_THINKING_BETA.to_string())]),
                );
                kwargs
            },
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
}

#[async_trait]
impl ChatProvider for Anthropic {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn thinking_effort(&self) -> Option<ThinkingEffort> {
        let thinking = self.generation_kwargs.get("thinking")?;
        let kind = thinking.get("type").and_then(|v| v.as_str())?;
        if kind == "disabled" {
            return Some(ThinkingEffort::Off);
        }
        let budget = thinking
            .get("budget_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if budget <= 1024 {
            Some(ThinkingEffort::Low)
        } else if budget <= 4096 {
            Some(ThinkingEffort::Medium)
        } else if budget <= 32000 {
            Some(ThinkingEffort::High)
        } else {
            Some(ThinkingEffort::XHigh)
        }
    }

    async fn generate(
        &self,
        system_prompt: &str,
        tools: &[Tool],
        history: &[Message],
    ) -> Result<Box<dyn StreamedMessage>, ChatProviderError> {
        let mut messages = Vec::new();
        for message in history {
            messages.push(convert_message(message)?);
        }
        mark_last_cacheable_message_block(&mut messages);

        let mut tool_defs = Vec::new();
        for tool in tools {
            tool_defs.push(convert_tool(tool));
        }
        mark_last_tool_definition(&mut tool_defs);

        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("messages".to_string(), Value::Array(messages));
        body.insert("tools".to_string(), Value::Array(tool_defs));
        body.insert("stream".to_string(), Value::Bool(self.stream));

        if !system_prompt.is_empty() {
            body.insert(
                "system".to_string(),
                json!([
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {
                            "type": "ephemeral",
                        }
                    }
                ]),
            );
        }

        let mut generation_kwargs = self.generation_kwargs.clone();
        let beta_header = extract_beta_header(generation_kwargs.remove("beta_features"));
        let extra_headers = generation_kwargs.remove("extra_headers");
        for (k, v) in generation_kwargs {
            body.insert(k, v);
        }

        let url = self
            .base_url
            .join("v1/messages")
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        let mut request = self.client.post(url);
        if !self.api_key.is_empty() {
            request = request.header("x-api-key", self.api_key.clone());
        }
        if let Some(beta_header) = beta_header {
            request = request.header("anthropic-beta", beta_header);
        }
        request = apply_extra_headers(request, extra_headers);

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
                format!("Anthropic API error ({status}): {text}"),
            ));
        }

        if self.stream {
            Ok(Box::new(AnthropicStreamedMessage::new_stream(resp)))
        } else {
            let value: Value = resp.json().await.map_err(map_reqwest_error)?;
            let (parts, message_id, usage) = parse_non_stream_response(&value)?;
            Ok(Box::new(AnthropicStreamedMessage::new_parts(
                parts, message_id, usage,
            )))
        }
    }

    fn with_thinking(&self, effort: ThinkingEffort) -> Box<dyn ChatProvider> {
        let thinking = match effort {
            ThinkingEffort::Off => json!({"type": "disabled"}),
            ThinkingEffort::Low => json!({"type": "enabled", "budget_tokens": 1024}),
            ThinkingEffort::Medium => json!({"type": "enabled", "budget_tokens": 4096}),
            ThinkingEffort::High => json!({"type": "enabled", "budget_tokens": 32000}),
            ThinkingEffort::XHigh => json!({"type": "enabled", "budget_tokens": 64000}),
        };

        let mut kwargs = Map::new();
        kwargs.insert("thinking".to_string(), thinking);

        let provider = self.clone().with_generation_kwargs(kwargs);
        Box::new(provider)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct AnthropicStreamedMessage {
    stream: Option<ByteStream>,
    buffer: String,
    event_lines: Vec<String>,
    parts: VecDeque<StreamedMessagePart>,
    id: Option<String>,
    usage: Option<TokenUsage>,
    tool_call_id_by_content_block_index: HashMap<i64, String>,
}

impl AnthropicStreamedMessage {
    pub fn new_stream(resp: reqwest::Response) -> Self {
        let stream = resp.bytes_stream();
        Self {
            stream: Some(Box::pin(stream)),
            buffer: String::new(),
            event_lines: Vec::new(),
            parts: VecDeque::new(),
            id: None,
            usage: None,
            tool_call_id_by_content_block_index: HashMap::new(),
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
            event_lines: Vec::new(),
            parts: parts.into(),
            id,
            usage,
            tool_call_id_by_content_block_index: HashMap::new(),
        }
    }

    fn process_event_block(&mut self) -> Result<bool, ChatProviderError> {
        if self.event_lines.is_empty() {
            return Ok(false);
        }

        let mut data_lines = Vec::new();
        for line in &self.event_lines {
            if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }
        self.event_lines.clear();

        if data_lines.is_empty() {
            return Ok(false);
        }

        let data = data_lines.join("\n");
        if data.trim() == "[DONE]" {
            return Ok(true);
        }

        let value: Value = serde_json::from_str(&data)
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        if let Some(err_value) = value.get("error") {
            let message = err_value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Anthropic stream error");
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Other,
                message.to_string(),
            ));
        }

        self.ingest_event(&value)
    }

    fn ingest_event(&mut self, value: &Value) -> Result<bool, ChatProviderError> {
        let event_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        match event_type {
            "message_start" => {
                if let Some(message) = value.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str()) {
                        self.id = Some(id.to_string());
                    }
                    if let Some(usage) = message.get("usage") {
                        merge_usage(&mut self.usage, usage);
                    }
                }
            }
            "message_delta" => {
                if let Some(usage) = value.get("usage") {
                    merge_usage(&mut self.usage, usage);
                }
            }
            "content_block_start" => {
                if let Some(block) = value.get("content_block") {
                    let block_index = value.get("index").and_then(|v| v.as_i64());
                    ingest_content_block_start(
                        block,
                        block_index,
                        &mut self.tool_call_id_by_content_block_index,
                        &mut self.parts,
                    )?;
                }
            }
            "content_block_delta" => {
                if let Some(delta) = value.get("delta") {
                    let block_index = value.get("index").and_then(|v| v.as_i64());
                    ingest_content_block_delta(
                        delta,
                        block_index,
                        &self.tool_call_id_by_content_block_index,
                        &mut self.parts,
                    );
                }
            }
            "message_stop" => {
                return Ok(true);
            }
            _ => {}
        }

        Ok(false)
    }
}

#[async_trait]
impl StreamedMessage for AnthropicStreamedMessage {
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
                        let line = self.buffer[..pos].trim_end_matches('\r').to_string();
                        self.buffer = self.buffer[pos + 1..].to_string();

                        if line.is_empty() {
                            if self.process_event_block()? {
                                self.stream = None;
                                break;
                            }
                            continue;
                        }

                        self.event_lines.push(line);
                    }
                }
                Some(Err(err)) => return Err(map_reqwest_error(err)),
                None => {
                    let _ = self.process_event_block()?;
                    self.stream = None;
                    continue;
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

fn convert_message(message: &Message) -> Result<Value, ChatProviderError> {
    match message.role {
        Role::System => Ok(json!({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": format!("<system>{}</system>", message.extract_text("\n")),
                }
            ]
        })),
        Role::Tool => {
            let tool_use_id = message.tool_call_id.clone().ok_or_else(|| {
                ChatProviderError::new(
                    ChatProviderErrorKind::Other,
                    "Tool message missing tool_call_id",
                )
            })?;
            let content = convert_tool_result_content(&message.content)?;
            Ok(json!({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": false,
                    }
                ]
            }))
        }
        Role::User | Role::Assistant => {
            let role = if matches!(message.role, Role::User) {
                "user"
            } else {
                "assistant"
            };

            let mut content = Vec::new();
            for part in &message.content {
                match part {
                    ContentPart::Text(text) => {
                        content.push(json!({
                            "type": "text",
                            "text": text.text,
                        }));
                    }
                    ContentPart::ImageUrl(image) => {
                        let source = image_url_to_anthropic_source(&image.image_url.url)?;
                        content.push(json!({
                            "type": "image",
                            "source": source,
                        }));
                    }
                    ContentPart::Think(think) => {
                        if let Some(signature) = &think.encrypted {
                            content.push(json!({
                                "type": "thinking",
                                "thinking": think.think,
                                "signature": signature,
                            }));
                        }
                    }
                    ContentPart::AudioUrl(_) | ContentPart::VideoUrl(_) => {}
                }
            }

            for tool_call in message.tool_calls.clone().unwrap_or_default() {
                let input = match tool_call.function.arguments {
                    Some(arguments) if !arguments.is_empty() => {
                        let parsed: Value = serde_json::from_str(&arguments).map_err(|_| {
                            ChatProviderError::new(
                                ChatProviderErrorKind::Other,
                                "Tool call arguments must be valid JSON",
                            )
                        })?;
                        if !parsed.is_object() {
                            return Err(ChatProviderError::new(
                                ChatProviderErrorKind::Other,
                                "Tool call arguments must be a JSON object",
                            ));
                        }
                        parsed
                    }
                    _ => json!({}),
                };

                content.push(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": input,
                }));
            }

            Ok(json!({
                "role": role,
                "content": content,
            }))
        }
    }
}

fn image_url_to_anthropic_source(url: &str) -> Result<Value, ChatProviderError> {
    if let Some(data) = url.strip_prefix("data:") {
        let (media_type, payload) = data.split_once(";base64,").ok_or_else(|| {
            ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Invalid data URL for image: {url}"),
            )
        })?;

        if !matches!(
            media_type,
            "image/png" | "image/jpeg" | "image/gif" | "image/webp"
        ) {
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Unsupported media type for base64 image: {media_type}, url: {url}"),
            ));
        }

        return Ok(json!({
            "type": "base64",
            "media_type": media_type,
            "data": payload,
        }));
    }

    Ok(json!({
        "type": "url",
        "url": url,
    }))
}

fn convert_tool_result_content(parts: &[ContentPart]) -> Result<Value, ChatProviderError> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) => {
                if !text.text.is_empty() {
                    blocks.push(json!({
                        "type": "text",
                        "text": text.text.clone(),
                    }));
                }
            }
            ContentPart::ImageUrl(image) => {
                let source = image_url_to_anthropic_source(&image.image_url.url)?;
                blocks.push(json!({
                    "type": "image",
                    "source": source,
                }));
            }
            other => {
                return Err(ChatProviderError::new(
                    ChatProviderErrorKind::Other,
                    format!("Anthropic API does not support {other:?} in tool result"),
                ));
            }
        }
    }
    Ok(Value::Array(blocks))
}

fn convert_tool(tool: &Tool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    })
}

fn parse_non_stream_response(value: &Value) -> Result<ParsedAnthropicResponse, ChatProviderError> {
    let message_id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let usage = value.get("usage").and_then(parse_usage);

    let mut parts = Vec::new();
    if let Some(content) = value.get("content").and_then(|v| v.as_array()) {
        for block in content {
            let kind = block
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            match kind {
                "text" => {
                    if let Some(text) = block
                        .get("text")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Text(
                            TextPart::new(text),
                        )));
                    }
                }
                "thinking" => {
                    if let Some(thinking) = block
                        .get("thinking")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(
                            ThinkPart {
                                kind: "think".to_string(),
                                think: thinking.to_string(),
                                encrypted: block
                                    .get("signature")
                                    .and_then(|v| v.as_str())
                                    .map(|v| v.to_string()),
                            },
                        )));
                    }
                }
                "redacted_thinking" => {
                    if let Some(signature) = block
                        .get("data")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(
                            ThinkPart {
                                kind: "think".to_string(),
                                think: String::new(),
                                encrypted: Some(signature.to_string()),
                            },
                        )));
                    }
                }
                "tool_use" => {
                    if let Some(tool_call) = parse_tool_use_block(block) {
                        parts.push(StreamedMessagePart::ToolCall(tool_call));
                    }
                }
                _ => {}
            }
        }
    }

    Ok((parts, message_id, usage))
}

fn ingest_content_block_start(
    block: &Value,
    block_index: Option<i64>,
    tool_call_id_by_content_block_index: &mut HashMap<i64, String>,
    parts: &mut VecDeque<StreamedMessagePart>,
) -> Result<(), ChatProviderError> {
    let kind = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match kind {
        "text" => {
            if let Some(text) = block
                .get("text")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Text(
                    TextPart::new(text),
                )));
            }
        }
        "thinking" => {
            if let Some(thinking) = block
                .get("thinking")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: thinking.to_string(),
                        encrypted: block
                            .get("signature")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_string()),
                    },
                )));
            }
        }
        "redacted_thinking" => {
            if let Some(signature) = block
                .get("data")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: String::new(),
                        encrypted: Some(signature.to_string()),
                    },
                )));
            }
        }
        "tool_use" => {
            if let Some(tool_call) = parse_tool_use_block_stream_start(block) {
                if let Some(block_index) = block_index {
                    tool_call_id_by_content_block_index.insert(block_index, tool_call.id.clone());
                }
                parts.push_back(StreamedMessagePart::ToolCall(tool_call));
            }
        }
        _ => {}
    }

    Ok(())
}

fn ingest_content_block_delta(
    delta: &Value,
    block_index: Option<i64>,
    tool_call_id_by_content_block_index: &HashMap<i64, String>,
    parts: &mut VecDeque<StreamedMessagePart>,
) {
    let delta_type = delta
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match delta_type {
        "text_delta" => {
            if let Some(text) = delta
                .get("text")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Text(
                    TextPart::new(text),
                )));
            }
        }
        "thinking_delta" => {
            if let Some(thinking) = delta
                .get("thinking")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: thinking.to_string(),
                        encrypted: None,
                    },
                )));
            }
        }
        "signature_delta" => {
            if let Some(signature) = delta
                .get("signature")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: String::new(),
                        encrypted: Some(signature.to_string()),
                    },
                )));
            }
        }
        "input_json_delta" => {
            if let Some(partial_json) = delta
                .get("partial_json")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                let tool_call_id = block_index
                    .and_then(|index| tool_call_id_by_content_block_index.get(&index).cloned());
                parts.push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                    arguments_part: Some(partial_json.to_string()),
                    tool_call_id,
                }));
            }
        }
        _ => {}
    }
}

fn parse_tool_use_block(block: &Value) -> Option<ToolCall> {
    let id = block.get("id").and_then(|v| v.as_str())?.to_string();
    let name = block.get("name").and_then(|v| v.as_str())?.to_string();

    let arguments = match block.get("input") {
        Some(input) => serde_json::to_string(input).ok(),
        None => Some("{}".to_string()),
    };

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction { name, arguments },
        extras: None,
    })
}

fn parse_tool_use_block_stream_start(block: &Value) -> Option<ToolCall> {
    let id = block.get("id").and_then(|v| v.as_str())?.to_string();
    let name = block.get("name").and_then(|v| v.as_str())?.to_string();

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction {
            name,
            arguments: None,
        },
        extras: None,
    })
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    Some(TokenUsage {
        input_other: value
            .get("input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        output: value
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        input_cache_read: value
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        input_cache_creation: value
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
    })
}

fn merge_usage(current: &mut Option<TokenUsage>, delta: &Value) {
    let input_other = delta.get("input_tokens").and_then(|v| v.as_i64());
    let output = delta.get("output_tokens").and_then(|v| v.as_i64());
    let input_cache_read = delta
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_i64());
    let input_cache_creation = delta
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_i64());

    if input_other.is_none()
        && output.is_none()
        && input_cache_read.is_none()
        && input_cache_creation.is_none()
    {
        return;
    }

    let usage = current.get_or_insert(TokenUsage {
        input_other: 0,
        output: 0,
        input_cache_read: 0,
        input_cache_creation: 0,
    });

    if let Some(value) = input_other {
        usage.input_other = value;
    }
    if let Some(value) = output {
        usage.output = value;
    }
    if let Some(value) = input_cache_read {
        usage.input_cache_read = value;
    }
    if let Some(value) = input_cache_creation {
        usage.input_cache_creation = value;
    }
}

fn extract_beta_header(beta_features: Option<Value>) -> Option<String> {
    match beta_features {
        Some(Value::String(value)) => {
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        }
        Some(Value::Array(values)) => {
            let mut features = Vec::new();
            for value in values {
                match value {
                    Value::String(value) if !value.is_empty() => features.push(value),
                    Value::Null => {}
                    other => features.push(other.to_string()),
                }
            }
            if features.is_empty() {
                None
            } else {
                Some(features.join(","))
            }
        }
        _ => None,
    }
}

fn apply_extra_headers(
    mut request: reqwest::RequestBuilder,
    extra_headers: Option<Value>,
) -> reqwest::RequestBuilder {
    if let Some(Value::Object(headers)) = extra_headers {
        for (name, value) in headers {
            let header_value = match value {
                Value::String(text) => text,
                Value::Null => continue,
                other => other.to_string(),
            };
            if header_value.is_empty() {
                continue;
            }
            request = request.header(name, header_value);
        }
    }
    request
}

fn mark_last_cacheable_message_block(messages: &mut [Value]) {
    let Some(last_message) = messages.last_mut() else {
        return;
    };
    let Some(content_blocks) = last_message
        .get_mut("content")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let Some(last_block) = content_blocks.last_mut() else {
        return;
    };
    let Some(block_type) = last_block.get("type").and_then(Value::as_str) else {
        return;
    };
    if !supports_anthropic_cache_control(block_type) {
        return;
    }
    if let Some(block) = last_block.as_object_mut() {
        block.insert(
            "cache_control".to_string(),
            json!({
                "type": "ephemeral",
            }),
        );
    }
}

fn mark_last_tool_definition(tool_defs: &mut [Value]) {
    let Some(last_tool) = tool_defs.last_mut() else {
        return;
    };
    if let Some(tool) = last_tool.as_object_mut() {
        tool.insert(
            "cache_control".to_string(),
            json!({
                "type": "ephemeral",
            }),
        );
    }
}

fn supports_anthropic_cache_control(block_type: &str) -> bool {
    matches!(
        block_type,
        "text"
            | "image"
            | "document"
            | "search_result"
            | "tool_use"
            | "tool_result"
            | "server_tool_use"
            | "web_search_tool_result"
    )
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
    fn test_parse_non_stream_response_extracts_content_and_tool_call() {
        let value = json!({
            "id": "msg-1",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 11,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 2
            },
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "thinking", "thinking": "reasoning", "signature": "enc" },
                { "type": "tool_use", "id": "call-1", "name": "sum", "input": { "a": 1 } }
            ]
        });

        let (parts, id, usage) = parse_non_stream_response(&value).unwrap();
        assert_eq!(id.as_deref(), Some("msg-1"));
        assert_eq!(usage.expect("usage").input_cache_read, 4);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_ingest_content_block_delta_supports_signature_and_tool_args() {
        let mut parts = VecDeque::new();
        let mut tool_call_id_by_content_block_index = HashMap::new();
        tool_call_id_by_content_block_index.insert(2, "call-1".to_string());
        ingest_content_block_delta(
            &json!({"type": "signature_delta", "signature": "enc"}),
            None,
            &tool_call_id_by_content_block_index,
            &mut parts,
        );
        ingest_content_block_delta(
            &json!({"type": "input_json_delta", "partial_json": "{\"a\":"}),
            Some(2),
            &tool_call_id_by_content_block_index,
            &mut parts,
        );
        assert_eq!(parts.len(), 2);
        if let Some(StreamedMessagePart::ToolCallPart(part)) = parts.get(1) {
            assert_eq!(part.tool_call_id.as_deref(), Some("call-1"));
        } else {
            panic!("expected second part to be ToolCallPart");
        }
    }

    #[test]
    fn test_merge_usage_preserves_existing_fields_for_partial_delta() {
        let mut usage = parse_usage(&json!({
            "input_tokens": 30,
            "output_tokens": 1,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2
        }));

        merge_usage(&mut usage, &json!({"output_tokens": 15}));

        let usage = usage.expect("usage");
        assert_eq!(usage.input_other, 30);
        assert_eq!(usage.output, 15);
        assert_eq!(usage.input_cache_read, 4);
        assert_eq!(usage.input_cache_creation, 2);
    }

    #[test]
    fn test_default_generation_kwargs_include_interleaved_thinking_beta() {
        let provider = Anthropic::new("claude-sonnet-4", None, None, None).expect("provider");
        let params = provider.model_parameters();
        assert_eq!(
            params.get("beta_features"),
            Some(&json!(["interleaved-thinking-2025-05-14"]))
        );
    }

    #[test]
    fn test_mark_last_cacheable_message_block_and_tool_definition() {
        let mut messages = vec![
            json!({
                "role": "user",
                "content": [{"type": "text", "text": "first"}]
            }),
            json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "sum", "input": {}}]
            }),
        ];
        mark_last_cacheable_message_block(&mut messages);
        assert_eq!(
            messages[1]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );

        let mut tools = vec![
            json!({"name": "a", "description": "a", "input_schema": {}}),
            json!({"name": "b", "description": "b", "input_schema": {}}),
        ];
        mark_last_tool_definition(&mut tools);
        assert_eq!(tools[1]["cache_control"], json!({"type": "ephemeral"}));
    }
}
