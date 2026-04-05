use axum::{
    body::Body,
    http::{Response, StatusCode},
};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info};

use crate::anthropic_types::{
    is_web_search_server_tool, parse_server_tool, ContentBlock, MessagesRequest, MessagesResponse,
    ServerTool,
};
use crate::auth::UpstreamAuth;
use crate::brave_search::{search_brave, BraveSearchParams};

const WEB_SEARCH_FUNCTION_NAME: &str = "__web_search";

pub struct WebSearchContext {
    pub enabled: bool,
    pub server_tool: Option<ServerTool>,
}

/// Preprocess an Anthropic Messages request to handle web search server tools.
/// Returns the modified request and web search context.
pub fn preprocess_web_search(
    mut payload: MessagesRequest,
    brave_api_key: &Option<String>,
) -> (MessagesRequest, WebSearchContext) {
    let tools = match &payload.tools {
        Some(tools) if !tools.is_empty() => tools,
        _ => {
            return (
                payload,
                WebSearchContext {
                    enabled: false,
                    server_tool: None,
                },
            );
        }
    };

    let mut server_tools = Vec::new();
    let mut regular_tools = Vec::new();

    for tool in tools {
        if is_web_search_server_tool(tool) {
            if let Some(st) = parse_server_tool(tool) {
                server_tools.push(st);
            }
        } else {
            regular_tools.push(tool.clone());
        }
    }

    if server_tools.is_empty() {
        return (
            payload,
            WebSearchContext {
                enabled: false,
                server_tool: None,
            },
        );
    }

    let server_tool = server_tools.into_iter().next().unwrap();

    if brave_api_key.is_none() {
        debug!("Web search requested but no Brave API key, stripping server tools");
        payload.tools = if regular_tools.is_empty() {
            None
        } else {
            Some(regular_tools)
        };
        return (
            payload,
            WebSearchContext {
                enabled: false,
                server_tool: None,
            },
        );
    }

    // Replace server tool with a regular function tool
    let web_search_function_tool = json!({
        "name": WEB_SEARCH_FUNCTION_NAME,
        "description": "Search the web for current information. Use this when you need up-to-date information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    });

    regular_tools.push(web_search_function_tool);
    payload.tools = Some(regular_tools);

    (
        payload,
        WebSearchContext {
            enabled: true,
            server_tool: Some(server_tool),
        },
    )
}

/// Generate a server tool use ID like the Anthropic API format.
fn generate_search_id() -> String {
    let uuid = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("srvtoolu_{}", &uuid[..24])
}

/// Execute web search and return formatted Anthropic blocks.
async fn execute_web_search(
    http_client: &reqwest::Client,
    api_key: &str,
    query: &str,
    server_tool: &Option<ServerTool>,
) -> (Value, Value) {
    let search_id = generate_search_id();

    let params = BraveSearchParams {
        query: query.to_string(),
        count: Some(5),
        allowed_domains: server_tool
            .as_ref()
            .and_then(|st| st.allowed_domains.clone()),
        blocked_domains: server_tool
            .as_ref()
            .and_then(|st| st.blocked_domains.clone()),
    };

    let results = match search_brave(http_client, api_key, &params).await {
        Ok(r) => r,
        Err(e) => {
            error!("Web search failed: {}", e);
            vec![crate::brave_search::BraveSearchResult {
                title: "Search failed".to_string(),
                url: String::new(),
                description: format!("Web search for \"{}\" failed. Please try again.", query),
                page_age: None,
            }]
        }
    };

    let server_tool_use = json!({
        "type": "server_tool_use",
        "id": search_id,
        "name": "web_search",
        "input": { "query": query }
    });

    let search_results: Vec<Value> = results
        .iter()
        .map(|r| {
            let mut result = json!({
                "type": "web_search_result",
                "url": r.url,
                "title": r.title,
                "encrypted_content": r.description
            });
            if let Some(ref page_age) = r.page_age {
                result["page_age"] = json!(page_age);
            }
            result
        })
        .collect();

    let tool_result = json!({
        "type": "web_search_tool_result",
        "tool_use_id": search_id,
        "content": search_results
    });

    (server_tool_use, tool_result)
}

/// Build a continuation payload with search results injected as tool result.
fn build_continuation_payload(
    original: &MessagesRequest,
    tool_call_id: &str,
    tool_call_name: &str,
    tool_call_input: &Value,
    search_results: &Value,
) -> MessagesRequest {
    let results_content = search_results["content"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|r| {
                    format!(
                        "[{}]({})\n{}",
                        r["title"].as_str().unwrap_or(""),
                        r["url"].as_str().unwrap_or(""),
                        r["encrypted_content"].as_str().unwrap_or("")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();

    let mut messages = original.messages.clone();

    // Append assistant message with tool_use
    messages.push(crate::anthropic_types::Message {
        role: "assistant".to_string(),
        content: json!([{
            "type": "tool_use",
            "id": tool_call_id,
            "name": tool_call_name,
            "input": tool_call_input
        }]),
    });

    // Append user message with tool_result
    messages.push(crate::anthropic_types::Message {
        role: "user".to_string(),
        content: json!([{
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": results_content
        }]),
    });

    MessagesRequest {
        model: original.model.clone(),
        messages,
        max_tokens: original.max_tokens,
        system: original.system.clone(),
        metadata: original.metadata.clone(),
        stop_sequences: original.stop_sequences.clone(),
        stream: original.stream,
        temperature: original.temperature,
        top_p: original.top_p,
        top_k: original.top_k,
        tools: None,
        tool_choice: None,
        thinking: original.thinking.clone(),
        service_tier: original.service_tier.clone(),
        extra: original.extra.clone(),
    }
}

/// Send a request to the upstream Claude API.
async fn send_to_upstream(
    http_client: &reqwest::Client,
    upstream_url: &str,
    upstream_auth: &Arc<dyn UpstreamAuth>,
    upstream_headers: &[(String, String)],
    body: &[u8],
) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
    let url = format!("{}/v1/messages", upstream_url);

    let auth_header = upstream_auth.get_auth_header().await?;
    let auth_header_name = upstream_auth.auth_header_name();

    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        HeaderName::from_bytes(auth_header_name.as_bytes())?,
        auth_header.clone(),
    );
    // Also add Authorization: Bearer for upstreams that require it
    if auth_header_name == "x-api-key" {
        let bearer_value = format!("Bearer {}", auth_header.to_str().unwrap_or(""));
        if let Ok(hv) = HeaderValue::from_str(&bearer_value) {
            headers
                .entry(HeaderName::from_static("authorization"))
                .or_insert(hv);
        }
    }

    // Add additional headers from auth provider
    if let Ok(additional) = upstream_auth.get_additional_headers().await {
        for (name, value) in additional {
            if let Ok(header_name) = HeaderName::try_from(name) {
                headers.insert(header_name, value);
            }
        }
    }

    // Add custom upstream headers
    for (name, value) in upstream_headers {
        if let (Ok(header_name), Ok(header_value)) = (
            HeaderName::try_from(name.as_str()),
            HeaderValue::from_str(value),
        ) {
            headers.insert(header_name, header_value);
        }
    }

    let response = http_client
        .post(&url)
        .headers(headers)
        .body(body.to_vec())
        .send()
        .await?;

    Ok(response)
}

/// Handle a non-streaming web search request.
pub async fn handle_non_streaming(
    payload: &MessagesRequest,
    context: &WebSearchContext,
    upstream_url: &str,
    upstream_auth: &Arc<dyn UpstreamAuth>,
    upstream_headers: &[(String, String)],
    http_client: &reqwest::Client,
    brave_api_key: &str,
) -> Result<Response<Body>, Response<Body>> {
    // Send modified request to upstream
    let body_bytes = serde_json::to_vec(payload).map_err(|e| {
        error!("Failed to serialize payload: {}", e);
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Serialization error: {}", e)))
            .unwrap()
    })?;

    let upstream_response = send_to_upstream(
        http_client,
        upstream_url,
        upstream_auth,
        upstream_headers,
        &body_bytes,
    )
    .await
    .map_err(|e| {
        error!("Upstream request failed: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Upstream request failed: {}", e)))
            .unwrap()
    })?;

    let status = upstream_response.status();
    let response_headers = upstream_response.headers().clone();
    let response_body = upstream_response.bytes().await.map_err(|e| {
        error!("Failed to read upstream response: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Failed to read response: {}", e)))
            .unwrap()
    })?;

    // If upstream returned an error, pass it through
    if !status.is_success() {
        let mut resp = Response::new(Body::from(response_body));
        *resp.status_mut() = status;
        for (name, value) in response_headers.iter() {
            resp.headers_mut().insert(name.clone(), value.clone());
        }
        return Ok(resp);
    }

    // Parse the response to check for web search tool calls
    let anthropic_response: MessagesResponse =
        serde_json::from_slice(&response_body).map_err(|e| {
            error!("Failed to parse upstream response: {}", e);
            // Return raw response if parsing fails
            let mut resp = Response::new(Body::from(response_body.clone()));
            *resp.status_mut() = status;
            resp
        })?;

    // Find __web_search tool call
    let web_search_call = anthropic_response.content.iter().find_map(|block| {
        if let ContentBlock::ToolUse { id, name, input } = block {
            if name == WEB_SEARCH_FUNCTION_NAME {
                return Some((id.clone(), input.clone()));
            }
        }
        None
    });

    let Some((tool_call_id, tool_call_input)) = web_search_call else {
        // No web search tool call, return original response
        let mut resp = Response::new(Body::from(response_body));
        *resp.status_mut() = status;
        for (name, value) in response_headers.iter() {
            resp.headers_mut().insert(name.clone(), value.clone());
        }
        return Ok(resp);
    };

    info!("Web search triggered (non-streaming)");

    let query = tool_call_input
        .get("query")
        .and_then(|q| q.as_str())
        .unwrap_or("");

    let (server_tool_use, tool_result) =
        execute_web_search(http_client, brave_api_key, query, &context.server_tool).await;

    // Build and send continuation request
    let mut continuation = build_continuation_payload(
        payload,
        &tool_call_id,
        WEB_SEARCH_FUNCTION_NAME,
        &tool_call_input,
        &tool_result,
    );
    continuation.stream = Some(false);

    let continuation_bytes = serde_json::to_vec(&continuation).map_err(|e| {
        error!("Failed to serialize continuation: {}", e);
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Serialization error: {}", e)))
            .unwrap()
    })?;

    let final_response = send_to_upstream(
        http_client,
        upstream_url,
        upstream_auth,
        upstream_headers,
        &continuation_bytes,
    )
    .await
    .map_err(|e| {
        error!("Continuation request failed: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Continuation request failed: {}", e)))
            .unwrap()
    })?;

    let final_status = final_response.status();
    let final_headers = final_response.headers().clone();
    let final_body = final_response.bytes().await.map_err(|e| {
        error!("Failed to read continuation response: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Failed to read response: {}", e)))
            .unwrap()
    })?;

    if !final_status.is_success() {
        let mut resp = Response::new(Body::from(final_body));
        *resp.status_mut() = final_status;
        for (name, value) in final_headers.iter() {
            resp.headers_mut().insert(name.clone(), value.clone());
        }
        return Ok(resp);
    }

    // Parse final response and prepend web search blocks
    let mut final_anthropic: MessagesResponse =
        serde_json::from_slice(&final_body).map_err(|e| {
            error!("Failed to parse continuation response: {}", e);
            let mut resp = Response::new(Body::from(final_body.clone()));
            *resp.status_mut() = final_status;
            resp
        })?;

    // Prepend server_tool_use and web_search_tool_result blocks
    let server_tool_use_block: ContentBlock = serde_json::from_value(server_tool_use).unwrap();
    let tool_result_block: ContentBlock = serde_json::from_value(tool_result).unwrap();

    let mut new_content = vec![server_tool_use_block, tool_result_block];
    new_content.append(&mut final_anthropic.content);
    final_anthropic.content = new_content;

    let result_json = serde_json::to_vec(&final_anthropic).map_err(|e| {
        error!("Failed to serialize final response: {}", e);
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Serialization error: {}", e)))
            .unwrap()
    })?;

    let mut resp = Response::new(Body::from(result_json));
    *resp.status_mut() = StatusCode::OK;
    resp.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    Ok(resp)
}

/// Handle a streaming web search request.
pub async fn handle_streaming(
    payload: &MessagesRequest,
    context: &WebSearchContext,
    upstream_url: &str,
    upstream_auth: &Arc<dyn UpstreamAuth>,
    upstream_headers: &[(String, String)],
    http_client: &reqwest::Client,
    brave_api_key: &str,
) -> Result<Response<Body>, Response<Body>> {
    // Send modified request (with stream=true) to upstream
    let body_bytes = serde_json::to_vec(payload).map_err(|e| {
        error!("Failed to serialize payload: {}", e);
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Serialization error: {}", e)))
            .unwrap()
    })?;

    let upstream_response = send_to_upstream(
        http_client,
        upstream_url,
        upstream_auth,
        upstream_headers,
        &body_bytes,
    )
    .await
    .map_err(|e| {
        error!("Upstream request failed: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Upstream request failed: {}", e)))
            .unwrap()
    })?;

    let status = upstream_response.status();

    // If upstream returned an error, pass it through
    if !status.is_success() {
        let error_body = upstream_response.bytes().await.unwrap_or_default();
        let mut resp = Response::new(Body::from(error_body));
        *resp.status_mut() = status;
        return Ok(resp);
    }

    // Buffer the entire SSE stream to detect web search tool calls
    let buffered = buffer_sse_stream(upstream_response).await;

    // Check if the model called __web_search
    let web_search_call = buffered
        .tool_calls
        .iter()
        .find(|tc| tc.name == WEB_SEARCH_FUNCTION_NAME);

    let Some(web_search_call) = web_search_call else {
        // No web search - replay buffered events as SSE
        return Ok(replay_buffered_as_sse(&buffered));
    };

    info!("Web search triggered (streaming)");

    let query = serde_json::from_str::<Value>(&web_search_call.arguments)
        .ok()
        .and_then(|v| v.get("query").and_then(|q| q.as_str()).map(String::from))
        .unwrap_or_default();

    let tool_call_id = web_search_call.id.clone();
    let tool_call_arguments = web_search_call.arguments.clone();

    let (server_tool_use, tool_result) =
        execute_web_search(http_client, brave_api_key, &query, &context.server_tool).await;

    // Build continuation payload
    let tool_call_input: Value =
        serde_json::from_str(&tool_call_arguments).unwrap_or(json!({"query": query}));
    let mut continuation = build_continuation_payload(
        payload,
        &tool_call_id,
        WEB_SEARCH_FUNCTION_NAME,
        &tool_call_input,
        &tool_result,
    );
    continuation.stream = Some(true);

    let continuation_bytes = serde_json::to_vec(&continuation).map_err(|e| {
        error!("Failed to serialize continuation: {}", e);
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(format!("Serialization error: {}", e)))
            .unwrap()
    })?;

    let continuation_response = send_to_upstream(
        http_client,
        upstream_url,
        upstream_auth,
        upstream_headers,
        &continuation_bytes,
    )
    .await
    .map_err(|e| {
        error!("Continuation request failed: {}", e);
        Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("Continuation request failed: {}", e)))
            .unwrap()
    })?;

    // Build the combined SSE response
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    let continuation_stream = continuation_response.bytes_stream();

    tokio::spawn(async move {
        // 1. Emit message_start
        let message_start = json!({
            "type": "message_start",
            "message": {
                "id": buffered.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": buffered.model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": buffered.input_tokens,
                    "output_tokens": 0
                }
            }
        });
        if emit_sse_event(&tx, "message_start", &message_start)
            .await
            .is_err()
        {
            return;
        }

        let mut block_index: usize = 0;

        // 2. Emit server_tool_use block
        if emit_block_with_delta(
            &tx,
            block_index,
            &json!({
                "type": "server_tool_use",
                "id": server_tool_use["id"],
                "name": server_tool_use["name"],
                "input": {}
            }),
            &json!({
                "type": "input_json_delta",
                "partial_json": serde_json::to_string(&server_tool_use["input"]).unwrap_or_default()
            }),
        )
        .await
        .is_err()
        {
            return;
        }
        block_index += 1;

        // 3. Emit web_search_tool_result block (full content in start, no delta)
        if emit_search_result_block(&tx, block_index, &tool_result)
            .await
            .is_err()
        {
            return;
        }
        block_index += 1;

        // 4. Stream continuation response
        let mut text_block_open = false;
        let mut byte_stream = continuation_stream;
        let mut line_buffer = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    error!("Continuation stream error: {}", e);
                    break;
                }
            };

            let chunk_str = String::from_utf8_lossy(&chunk);
            line_buffer.push_str(&chunk_str);

            // Process complete SSE lines
            while let Some(line_end) = line_buffer.find('\n') {
                let line = line_buffer[..line_end].trim().to_string();
                line_buffer = line_buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        // Close text block if open, then end
                        if text_block_open {
                            let _ = emit_sse_event(
                                &tx,
                                "content_block_stop",
                                &json!({ "type": "content_block_stop", "index": block_index }),
                            )
                            .await;
                        }
                        break;
                    }

                    if let Ok(event) = serde_json::from_str::<Value>(data) {
                        let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match event_type {
                            "content_block_start" => {
                                // Rewrite the index
                                let mut rewritten = event.clone();
                                rewritten["index"] = json!(block_index);
                                text_block_open = true;
                                let _ =
                                    emit_sse_event(&tx, "content_block_start", &rewritten).await;
                            }
                            "content_block_delta" => {
                                let mut rewritten = event.clone();
                                rewritten["index"] = json!(block_index);
                                let _ =
                                    emit_sse_event(&tx, "content_block_delta", &rewritten).await;
                            }
                            "content_block_stop" => {
                                let _ = emit_sse_event(
                                    &tx,
                                    "content_block_stop",
                                    &json!({ "type": "content_block_stop", "index": block_index }),
                                )
                                .await;
                                text_block_open = false;
                                block_index += 1;
                            }
                            "message_delta" => {
                                // Forward message_delta as-is
                                let _ = emit_sse_event(&tx, "message_delta", &event).await;
                            }
                            "message_stop" => {
                                let _ = emit_sse_event(
                                    &tx,
                                    "message_stop",
                                    &json!({ "type": "message_stop" }),
                                )
                                .await;
                            }
                            "ping" => {
                                let _ = emit_sse_event(&tx, "ping", &json!({"type": "ping"})).await;
                            }
                            "message_start" => {
                                // Skip the continuation's message_start, we already emitted ours
                            }
                            _ => {
                                // Forward unknown events
                                let _ = emit_sse_event(&tx, event_type, &event).await;
                            }
                        }
                    }
                }
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("cache-control"),
        HeaderValue::from_static("no-cache"),
    );
    Ok(response)
}

// SSE helper functions

async fn emit_sse_event(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    event_type: &str,
    data: &Value,
) -> Result<(), ()> {
    let event_str = format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_default()
    );
    tx.send(Ok(Bytes::from(event_str))).await.map_err(|_| ())
}

async fn emit_block_with_delta(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    index: usize,
    content_block: &Value,
    delta: &Value,
) -> Result<(), ()> {
    // content_block_start
    emit_sse_event(
        tx,
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block
        }),
    )
    .await?;

    // content_block_delta
    emit_sse_event(
        tx,
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": delta
        }),
    )
    .await?;

    // content_block_stop
    emit_sse_event(
        tx,
        "content_block_stop",
        &json!({
            "type": "content_block_stop",
            "index": index
        }),
    )
    .await?;

    Ok(())
}

async fn emit_search_result_block(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    index: usize,
    tool_result: &Value,
) -> Result<(), ()> {
    // Emit full content in content_block_start
    emit_sse_event(
        tx,
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_result["tool_use_id"],
                "content": tool_result["content"]
            }
        }),
    )
    .await?;

    emit_sse_event(
        tx,
        "content_block_stop",
        &json!({
            "type": "content_block_stop",
            "index": index
        }),
    )
    .await?;

    Ok(())
}

// Buffering logic for SSE streams

struct BufferedToolCall {
    id: String,
    name: String,
    arguments: String,
}

struct BufferedStreamResponse {
    id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    tool_calls: Vec<BufferedToolCall>,
    raw_events: Vec<String>,
}

async fn buffer_sse_stream(response: reqwest::Response) -> BufferedStreamResponse {
    let mut result = BufferedStreamResponse {
        id: String::new(),
        model: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        tool_calls: Vec::new(),
        raw_events: Vec::new(),
    };

    // Track tool calls by index for incremental assembly
    let mut tool_call_map: std::collections::HashMap<usize, BufferedToolCall> =
        std::collections::HashMap::new();

    let mut byte_stream = response.bytes_stream();
    let mut line_buffer = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        let chunk = match chunk_result {
            Ok(b) => b,
            Err(_) => break,
        };

        let chunk_str = String::from_utf8_lossy(&chunk);
        line_buffer.push_str(&chunk_str);

        while let Some(line_end) = line_buffer.find('\n') {
            let line = line_buffer[..line_end].trim().to_string();
            line_buffer = line_buffer[line_end + 1..].to_string();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            // Store raw event for replay
            result.raw_events.push(line.clone());

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    result.raw_events.push("data: [DONE]".to_string());
                    break;
                }

                if let Ok(event) = serde_json::from_str::<Value>(data) {
                    let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match event_type {
                        "message_start" => {
                            if let Some(message) = event.get("message") {
                                result.id = message
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                result.model = message
                                    .get("model")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if let Some(usage) = message.get("usage") {
                                    result.input_tokens = usage
                                        .get("input_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                }
                            }
                        }
                        "content_block_start" => {
                            if let Some(block) = event.get("content_block") {
                                let block_type =
                                    block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                if block_type == "tool_use" {
                                    let index =
                                        event.get("index").and_then(|i| i.as_u64()).unwrap_or(0)
                                            as usize;
                                    let id = block
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = block
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    tool_call_map.insert(
                                        index,
                                        BufferedToolCall {
                                            id,
                                            name,
                                            arguments: String::new(),
                                        },
                                    );
                                }
                            }
                        }
                        "content_block_delta" => {
                            let index =
                                event.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            if let Some(tc) = tool_call_map.get_mut(&index) {
                                if let Some(delta) = event.get("delta") {
                                    if let Some(partial) =
                                        delta.get("partial_json").and_then(|v| v.as_str())
                                    {
                                        tc.arguments.push_str(partial);
                                    }
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(usage) = event.get("usage") {
                                result.output_tokens = usage
                                    .get("output_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(result.output_tokens);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    result.tool_calls = tool_call_map.into_values().collect();
    result
}

/// Replay buffered SSE events as-is (when no web search was needed).
fn replay_buffered_as_sse(buffered: &BufferedStreamResponse) -> Response<Body> {
    let mut body_str = String::new();
    for event in &buffered.raw_events {
        body_str.push_str(event);
        body_str.push('\n');
        // If this is a data line, add the extra blank line for SSE
        if event.starts_with("data: ") || event.starts_with("event: ") {
            // Only add blank line after data lines (SSE requires blank line between events)
        }
    }

    let mut response = Response::new(Body::from(body_str));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("cache-control"),
        HeaderValue::from_static("no-cache"),
    );
    response
}
