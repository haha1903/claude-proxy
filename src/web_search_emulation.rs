//! Web search emulation.
//!
//! Intercepts Anthropic Messages API requests whose ONLY tool is `web_search`,
//! runs the search through the configured providers, and synthesizes an
//! Anthropic-format response (server_tool_use + web_search_tool_result + text)
//! WITHOUT calling the upstream LLM. Ported from sub2api's
//! `gateway_websearch_emulation.go`.

use axum::{
    body::Body,
    http::{header, Response, StatusCode},
};
use serde_json::{json, Value};
use std::fmt::Write as _;
use tracing::{info, warn};

use crate::websearch::{SearchRequest, SearchResult, WebSearchManager, DEFAULT_MAX_RESULTS};

/// Tool names recognized as a web search request.
const WEB_SEARCH_TOOL_NAMES: &[&str] = &["web_search", "google_search", "web_search_20250305"];

/// Fallback model name echoed back when the request omits one.
const DEFAULT_MODEL: &str = "claude-sonnet-4-6";

/// Rough token estimate: ~4 chars per token.
const TOKEN_ESTIMATE_DIVISOR: usize = 4;

/// Check whether the request body contains exactly one tool and that tool is a
/// web search tool. Returns false on any parse failure (so the caller forwards
/// the request unchanged).
pub fn is_only_web_search_request(body: &[u8]) -> bool {
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let tools = match parsed.get("tools").and_then(|t| t.as_array()) {
        Some(arr) => arr,
        None => return false,
    };
    if tools.len() != 1 {
        return false;
    }
    is_web_search_tool(&tools[0])
}

/// A tool JSON value is a web search tool if its `type` starts with
/// "web_search" / equals "google_search", or its `name` is a known search name.
fn is_web_search_tool(tool: &Value) -> bool {
    if let Some(tool_type) = tool.get("type").and_then(|t| t.as_str()) {
        if tool_type.starts_with("web_search") || tool_type == "google_search" {
            return true;
        }
    }
    if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
        if WEB_SEARCH_TOOL_NAMES.contains(&name) {
            return true;
        }
    }
    false
}

/// Extract the search query: the text of the last user message.
fn extract_query(body: &Value) -> Option<String> {
    let messages = body.get("messages")?.as_array()?;
    let last = messages.last()?;
    if last.get("role").and_then(|r| r.as_str()) != Some("user") {
        return None;
    }
    let content = last.get("content")?;
    extract_text_from_content(content)
}

/// Pull the first usable text out of a message `content` (string or block array).
fn extract_text_from_content(content: &Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        if !s.is_empty() {
            return Some(s.to_string());
        }
        return None;
    }
    if let Some(blocks) = content.as_array() {
        for block in blocks {
            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        return Some(text.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Read the request's `model`, falling back to a default.
fn extract_model(body: &Value) -> String {
    body.get("model")
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

/// Read the request's `stream` flag (default false).
fn is_streaming(body: &Value) -> bool {
    body.get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false)
}

/// Handle a web-search-only request: search and synthesize an Anthropic response.
/// `body` is the buffered request body (already confirmed to be web-search-only).
pub async fn handle(manager: &WebSearchManager, body: &[u8]) -> Response<Body> {
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid JSON: {}", e)),
    };

    let model = extract_model(&parsed);
    let streaming = is_streaming(&parsed);

    let query = match extract_query(&parsed) {
        Some(q) => q,
        None => {
            warn!("web search emulation: no query found in messages");
            return error_response(StatusCode::BAD_REQUEST, "no search query found in messages");
        }
    };

    info!(
        "web search emulation: executing search for query: {}",
        query
    );

    let req = SearchRequest {
        query: query.clone(),
        max_results: DEFAULT_MAX_RESULTS,
    };

    // On total failure, degrade gracefully: synthesize a well-formed response
    // whose text explains the search failed, so the client still gets valid output.
    let results = match manager.search(&req).await {
        Some((resp, provider)) => {
            info!(
                "web search emulation: search completed via '{}', {} results",
                provider,
                resp.results.len()
            );
            resp.results
        }
        None => {
            warn!(
                "web search emulation: all providers failed for query: {}",
                query
            );
            Vec::new()
        }
    };

    let summary = build_text_summary(&query, &results);
    let synthesized = Synthesized {
        message_id: format!("msg_ws_{}", uuid::Uuid::new_v4().simple()),
        tool_use_id: format!("srvtoolu_ws_{}", short_uuid()),
        model,
        query,
        results,
        summary,
    };

    if streaming {
        synthesized.into_sse_response()
    } else {
        synthesized.into_json_response()
    }
}

/// A synthesized web search result, ready to render as JSON or SSE.
struct Synthesized {
    message_id: String,
    tool_use_id: String,
    model: String,
    query: String,
    results: Vec<SearchResult>,
    summary: String,
}

/// A 16-char hex fragment for the tool-use id, matching the Anthropic id shape.
fn short_uuid() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..16].to_string()
}

/// Build the `web_search_result` content blocks from search results.
fn build_result_blocks(results: &[SearchResult]) -> Vec<Value> {
    results
        .iter()
        .map(|r| {
            let mut block = json!({
                "type": "web_search_result",
                "url": r.url,
                "title": r.title,
            });
            if !r.snippet.is_empty() {
                block["page_content"] = json!(r.snippet);
            }
            if let Some(ref page_age) = r.page_age {
                block["page_age"] = json!(page_age);
            }
            block
        })
        .collect()
}

/// Build the human-readable markdown summary text block.
fn build_text_summary(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No search results found for: {}", query);
    }
    let mut sb = String::new();
    let _ = write!(sb, "Here are the search results for \"{}\":\n\n", query);
    for (i, r) in results.iter().enumerate() {
        let _ = write!(
            sb,
            "{}. **{}**\n   {}\n   {}\n\n",
            i + 1,
            r.title,
            r.url,
            r.snippet
        );
    }
    sb
}

impl Synthesized {
    fn output_tokens(&self) -> usize {
        self.summary.len() / TOKEN_ESTIMATE_DIVISOR
    }

    fn server_tool_use_block(&self) -> Value {
        json!({ "type": "server_tool_use", "id": self.tool_use_id, "name": "web_search", "input": { "query": self.query } })
    }

    fn tool_result_block(&self) -> Value {
        json!({ "type": "web_search_tool_result", "tool_use_id": self.tool_use_id, "content": build_result_blocks(&self.results) })
    }

    /// Build a non-streaming JSON message response.
    fn into_json_response(self) -> Response<Body> {
        let msg = json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [
                self.server_tool_use_block(),
                self.tool_result_block(),
                { "type": "text", "text": self.summary },
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 0, "output_tokens": self.output_tokens() },
        });

        match serde_json::to_vec(&msg) {
            Ok(body) => Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap(),
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to serialize response: {}", e),
            ),
        }
    }

    /// Build a streaming SSE response. All data is already available, so the
    /// full event sequence is assembled up front and returned as a single body.
    fn into_sse_response(self) -> Response<Body> {
        let mut sse = String::new();

        push_sse_event(
            &mut sse,
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id, "type": "message", "role": "assistant", "model": self.model,
                    "content": [], "stop_reason": null, "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                },
            }),
        );

        push_content_block(&mut sse, 0, self.server_tool_use_block());
        push_content_block(&mut sse, 1, self.tool_result_block());

        // index 2: text summary, streamed as a single delta
        push_sse_event(
            &mut sse,
            "content_block_start",
            &json!({ "type": "content_block_start", "index": 2, "content_block": { "type": "text", "text": "" } }),
        );
        push_sse_event(
            &mut sse,
            "content_block_delta",
            &json!({ "type": "content_block_delta", "index": 2, "delta": { "type": "text_delta", "text": self.summary } }),
        );
        push_sse_event(
            &mut sse,
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": 2 }),
        );

        push_sse_event(
            &mut sse,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                "usage": { "output_tokens": self.output_tokens() },
            }),
        );
        push_sse_event(&mut sse, "message_stop", &json!({ "type": "message_stop" }));

        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(sse))
            .unwrap()
    }
}

/// Emit a complete `content_block_start` + `content_block_stop` pair for a block
/// whose content is fully known up front (no deltas).
fn push_content_block(buf: &mut String, index: u32, block: Value) {
    push_sse_event(
        buf,
        "content_block_start",
        &json!({ "type": "content_block_start", "index": index, "content_block": block }),
    );
    push_sse_event(
        buf,
        "content_block_stop",
        &json!({ "type": "content_block_stop", "index": index }),
    );
}

/// Append one SSE event (`event:` + `data:` lines) to the buffer.
fn push_sse_event(buf: &mut String, event: &str, data: &Value) {
    let _ = write!(buf, "event: {}\ndata: {}\n\n", event, data);
}

/// Build a plain error response.
fn error_response(status: StatusCode, msg: &str) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::from(msg.to_string()))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_web_search_by_type() {
        let body = br#"{"tools":[{"type":"web_search_20250305","name":"web_search"}]}"#;
        assert!(is_only_web_search_request(body));
    }

    #[test]
    fn detects_web_search_by_name_only() {
        let body = br#"{"tools":[{"name":"web_search"}]}"#;
        assert!(is_only_web_search_request(body));
    }

    #[test]
    fn detects_google_search() {
        let body = br#"{"tools":[{"type":"google_search"}]}"#;
        assert!(is_only_web_search_request(body));
    }

    #[test]
    fn rejects_when_multiple_tools() {
        let body = br#"{"tools":[{"type":"web_search_20250305"},{"name":"bash"}]}"#;
        assert!(!is_only_web_search_request(body));
    }

    #[test]
    fn rejects_non_web_search_tool() {
        let body = br#"{"tools":[{"name":"bash"}]}"#;
        assert!(!is_only_web_search_request(body));
    }

    #[test]
    fn rejects_no_tools() {
        assert!(!is_only_web_search_request(br#"{"messages":[]}"#));
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(!is_only_web_search_request(b"not json"));
    }

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn extracts_query_from_string_content() {
        let body = parse(r#"{"messages":[{"role":"user","content":"what is rust"}]}"#);
        assert_eq!(extract_query(&body).as_deref(), Some("what is rust"));
    }

    #[test]
    fn extracts_query_from_block_content() {
        let body =
            parse(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}"#);
        assert_eq!(extract_query(&body).as_deref(), Some("hello"));
    }

    #[test]
    fn no_query_when_last_message_not_user() {
        let body = parse(r#"{"messages":[{"role":"assistant","content":"hi"}]}"#);
        assert_eq!(extract_query(&body), None);
    }

    #[test]
    fn model_defaults_when_absent() {
        assert_eq!(extract_model(&parse("{}")), DEFAULT_MODEL);
        assert_eq!(extract_model(&parse(r#"{"model":"claude-x"}"#)), "claude-x");
    }

    #[test]
    fn summary_lists_results() {
        let results = vec![SearchResult {
            url: "https://go.dev".to_string(),
            title: "Go".to_string(),
            snippet: "Go lang".to_string(),
            page_age: None,
        }];
        let s = build_text_summary("golang", &results);
        assert!(s.contains("golang"));
        assert!(s.contains("https://go.dev"));
        assert!(s.contains("**Go**"));
    }

    #[test]
    fn summary_handles_empty_results() {
        let s = build_text_summary("nothing", &[]);
        assert!(s.contains("No search results"));
    }

    #[test]
    fn result_blocks_include_page_age_only_when_present() {
        let results = vec![
            SearchResult {
                url: "u1".to_string(),
                title: "t1".to_string(),
                snippet: "s1".to_string(),
                page_age: Some("1 day".to_string()),
            },
            SearchResult {
                url: "u2".to_string(),
                title: "t2".to_string(),
                snippet: String::new(),
                page_age: None,
            },
        ];
        let blocks = build_result_blocks(&results);
        assert_eq!(blocks[0]["page_age"], json!("1 day"));
        assert_eq!(blocks[0]["page_content"], json!("s1"));
        assert!(blocks[1].get("page_age").is_none());
        // empty snippet omits page_content
        assert!(blocks[1].get("page_content").is_none());
    }
}
