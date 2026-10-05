use axum::{
    body::Body,
    extract::State,
    http::{header, Request, Response, StatusCode},
    response::IntoResponse,
};
use bytes::Bytes;
use futures::StreamExt;
use http_body_util::BodyStream;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

use crate::copilot_responses::ResponsesStream;
use crate::middleware::SelectedUpstreamAuth;
use crate::web_search_emulation;
use crate::websearch::WebSearchManager;

/// Shared state for the proxy
#[derive(Clone)]
pub struct ProxyState {
    pub upstream_url: String,
    pub http_client: Arc<RwLock<reqwest::Client>>,
    pub upstream_headers: Vec<(String, String)>,
    /// Web search manager; `None` disables emulation (pure passthrough).
    pub web_search: Option<Arc<WebSearchManager>>,
    pub copilot_responses: bool,
}

/// Build a reqwest client for upstream requests.
pub fn build_http_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder().build()
}

impl ProxyState {
    async fn http_client(&self) -> reqwest::Client {
        self.http_client.read().await.clone()
    }

    async fn rotate_http_client(&self) {
        match build_http_client() {
            Ok(new_client) => {
                *self.http_client.write().await = new_client;
                warn!("Replaced upstream HTTP client after upstream 500 response");
            }
            Err(e) => {
                error!("Failed to replace upstream HTTP client: {}", e);
            }
        }
    }
}

/// Headers that should not be forwarded to the upstream
const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
];

/// Header prefixes that should not be forwarded (proxy/infrastructure headers)
const STRIPPED_HEADER_PREFIXES: &[&str] = &["x-ms-", "x-forwarded-", "x-k8se-", "x-envoy-"];

/// Check if a header should be stripped (hop-by-hop or infrastructure header)
fn should_strip_header(name: &str) -> bool {
    if HOP_BY_HOP_HEADERS.contains(&name) {
        return true;
    }
    for prefix in STRIPPED_HEADER_PREFIXES {
        if name.starts_with(prefix) {
            return true;
        }
    }
    false
}

/// Authentication-related headers
const AUTH_HEADERS: &[&str] = &["authorization", "api-key", "x-api-key"];

const UPSTREAM_500_RESPONSE_DELAY_SECS: u64 = 30;

fn is_responses_request(method: &reqwest::Method, target: &str) -> bool {
    method == reqwest::Method::POST
        && matches!(
            target.split('?').next(),
            Some("/responses" | "/v1/responses")
        )
}

/// Proxy handler that forwards requests to the Claude API
pub async fn proxy_handler(
    State(state): State<ProxyState>,
    request: Request<Body>,
) -> impl IntoResponse {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path();
    let query = uri.query().map(|q| format!("?{}", q)).unwrap_or_default();
    let request_target = format!("{}{}", path, query);

    info!("Proxying {} {}", method, request_target);

    // Log request details at debug level
    debug!("Request headers:");
    for (name, value) in request.headers() {
        let name_str = name.as_str().to_lowercase();
        // Mask sensitive headers
        if AUTH_HEADERS.contains(&name_str.as_str()) {
            debug!("  {}: [REDACTED]", name);
        } else {
            debug!("  {}: {:?}", name, value);
        }
    }

    // Build the upstream URL
    let upstream_url = format!("{}{}", state.upstream_url, request_target);

    // Check if the client is authenticated
    let selected_auth = request.extensions().get::<SelectedUpstreamAuth>().cloned();
    let client_authenticated = selected_auth.is_some();

    // Build headers for upstream request
    let mut upstream_headers = HeaderMap::new();

    // Only add upstream authentication if the client provided a valid API key
    if let Some(SelectedUpstreamAuth(upstream_auth)) = selected_auth {
        // Get auth header for upstream
        let auth_header = match upstream_auth.get_auth_header().await {
            Ok(header) => header,
            Err(e) => {
                error!("Failed to get upstream auth header: {}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!("Authentication error: {}", e)))
                    .unwrap();
            }
        };

        let auth_header_name = upstream_auth.auth_header_name();
        let is_api_key_auth = auth_header_name == "x-api-key";

        // Copy headers from original request based on auth type
        for (name, value) in request.headers() {
            let name_str = name.as_str().to_lowercase();

            // Skip hop-by-hop headers
            if should_strip_header(&name_str) {
                continue;
            }

            // Handle auth headers based on upstream auth type
            if AUTH_HEADERS.contains(&name_str.as_str()) {
                if is_api_key_auth {
                    // For API key auth: replace auth headers with upstream API key
                    // Authorization header uses "Bearer <key>" format
                    // api-key and x-api-key use the raw key value
                    if name_str == "authorization" {
                        let bearer_value = format!("Bearer {}", auth_header.to_str().unwrap_or(""));
                        if let Ok(hv) = HeaderValue::from_str(&bearer_value) {
                            upstream_headers.insert(name.clone(), hv);
                        }
                    } else {
                        // api-key or x-api-key: use raw key value
                        upstream_headers.insert(name.clone(), auth_header.clone());
                    }
                }
                // For bearer auth (AAD/AzCli): skip client auth headers (they'll be replaced)
            } else {
                upstream_headers.insert(name.clone(), value.clone());
            }
        }

        // For bearer auth (AAD/AzCli): add the authorization header
        if !is_api_key_auth {
            upstream_headers.insert(HeaderName::from_static("authorization"), auth_header);
        }

        // Get additional headers from auth provider
        match upstream_auth.get_additional_headers().await {
            Ok(additional) => {
                for (name, value) in additional {
                    if let Ok(header_name) = HeaderName::try_from(name) {
                        upstream_headers.insert(header_name, value);
                    }
                }
            }
            Err(e) => {
                error!("Failed to get additional auth headers: {}", e);
            }
        }
    } else {
        // Passthrough mode: copy all headers except hop-by-hop
        debug!("Relaying request without upstream authentication");
        for (name, value) in request.headers() {
            let name_str = name.as_str().to_lowercase();
            if !should_strip_header(&name_str) {
                upstream_headers.insert(name.clone(), value.clone());
            }
        }
    }

    // Add custom upstream headers
    for (name, value) in &state.upstream_headers {
        if let (Ok(header_name), Ok(header_value)) = (
            HeaderName::try_from(name.as_str()),
            HeaderValue::from_str(value),
        ) {
            debug!("Adding custom header: {}: {}", name, value);
            upstream_headers.insert(header_name, header_value);
        }
    }

    // Log content-length if present
    if let Some(content_length) = upstream_headers.get("content-length") {
        debug!(
            "Request body size: {} bytes",
            content_length.to_str().unwrap_or("unknown")
        );
    }

    // Web search emulation: only for authenticated /v1/messages when a manager
    // is configured. The body must be buffered to inspect the tools, so this
    // branch is entered only when web search is enabled — the default path below
    // keeps streaming without buffering.
    if client_authenticated && path == "/v1/messages" {
        if let Some(web_search) = state.web_search.clone() {
            return handle_messages_with_web_search(
                &state,
                web_search,
                request,
                method,
                &upstream_url,
                &request_target,
                upstream_headers,
            )
            .await;
        }
    }

    if state.copilot_responses && is_responses_request(&method, &request_target) {
        upstream_headers.insert(
            header::ACCEPT_ENCODING,
            HeaderValue::from_static("identity"),
        );
    }

    // Stream the request body directly to upstream without buffering
    let request_body = request.into_body();
    let body_stream = BodyStream::new(request_body);
    let reqwest_body = reqwest::Body::wrap_stream(body_stream.map(|result| {
        result
            .map(|frame| frame.into_data().unwrap_or_default())
            .map_err(|e| std::io::Error::other(e.to_string()))
    }));

    // Build and send the upstream request
    let http_client = state.http_client().await;
    let upstream_request = http_client
        .request(method.clone(), &upstream_url)
        .headers(upstream_headers)
        .body(reqwest_body);

    relay_upstream(
        &state,
        &method,
        &request_target,
        http_client,
        upstream_request,
    )
    .await
}

/// Send an upstream request and relay its response back to the client,
/// streaming the body. Applies the upstream-500 → 503 back-off behavior.
/// Shared by the streaming passthrough path and the buffered web-search-forward
/// path so both have identical upstream semantics.
async fn relay_upstream(
    state: &ProxyState,
    method: &reqwest::Method,
    request_target: &str,
    http_client: reqwest::Client,
    upstream_request: reqwest::RequestBuilder,
) -> Response<Body> {
    let upstream_response = match upstream_request.send().await {
        Ok(response) => response,
        Err(e) => {
            error!("Upstream request failed: {}", e);
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("Upstream request failed: {}", e)))
                .unwrap();
        }
    };

    // Build response headers
    let status = upstream_response.status();
    if let Some(reason) = status.canonical_reason() {
        info!("Upstream response: {} {}", status.as_u16(), reason);
    } else {
        info!("Upstream response: {}", status.as_u16());
    }

    if status == StatusCode::INTERNAL_SERVER_ERROR {
        warn!(
            "Upstream returned 500 for {} {}; dropping upstream response body, rotating upstream HTTP client, delaying {} seconds, returning 503, and closing downstream connection",
            method, request_target, UPSTREAM_500_RESPONSE_DELAY_SECS
        );
        drop(upstream_response);
        drop(http_client);
        state.rotate_http_client().await;
        tokio::time::sleep(Duration::from_secs(UPSTREAM_500_RESPONSE_DELAY_SECS)).await;

        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header(header::CONNECTION, HeaderValue::from_static("close"))
            .body(Body::empty())
            .unwrap();
    }

    let mut response_headers = HeaderMap::new();

    for (name, value) in upstream_response.headers() {
        let name_str = name.as_str().to_lowercase();
        if !should_strip_header(&name_str) {
            response_headers.insert(name.clone(), value.clone());
        }
    }

    let normalize = state.copilot_responses
        && is_responses_request(method, request_target)
        && status.is_success()
        && response_headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("text/event-stream")
            })
        && response_headers
            .get(header::CONTENT_ENCODING)
            .is_none_or(|value| value == "identity");
    if normalize {
        for name in [
            "content-length",
            "etag",
            "content-md5",
            "digest",
            "content-digest",
            "repr-digest",
        ] {
            response_headers.remove(name);
        }
    }

    // Only Copilot Responses buffers individual SSE events for ID normalization.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    let mut byte_stream = upstream_response.bytes_stream();
    tokio::spawn(async move {
        let mut normalizer = normalize.then(ResponsesStream::default);
        loop {
            let chunk = tokio::select! {
                biased;
                _ = tx.closed() => break,
                chunk = byte_stream.next() => chunk,
            };
            let Some(chunk) = chunk else {
                if let Some(normalizer) = &mut normalizer {
                    let tail = normalizer.finish();
                    if !tail.is_empty() {
                        let _ = tx.send(Ok(tail)).await;
                    }
                }
                break;
            };
            match chunk {
                Ok(bytes) => {
                    let frames = match &mut normalizer {
                        Some(normalizer) => normalizer.push(bytes),
                        None => vec![bytes],
                    };
                    for frame in frames {
                        if tx.send(Ok(frame)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    if let Some(normalizer) = &mut normalizer {
                        let tail = normalizer.drain();
                        if !tail.is_empty() {
                            let _ = tx.send(Ok(tail)).await;
                        }
                    }
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

/// Handle an authenticated `/v1/messages` request when web search is enabled.
/// Buffers the body to inspect tools: if it is a web-search-only request, serve
/// it via emulation; otherwise forward the buffered body to the upstream as-is.
async fn handle_messages_with_web_search(
    state: &ProxyState,
    web_search: Arc<WebSearchManager>,
    request: Request<Body>,
    method: reqwest::Method,
    upstream_url: &str,
    request_target: &str,
    upstream_headers: HeaderMap,
) -> Response<Body> {
    use http_body_util::BodyExt;

    // Buffer the entire request body so we can inspect it.
    let body_bytes = match request.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(format!("Failed to read request body: {}", e)))
                .unwrap();
        }
    };

    if web_search_emulation::is_only_web_search_request(&body_bytes) {
        debug!("Intercepting web-search-only request for emulation");
        return web_search_emulation::handle(&web_search, &body_bytes).await;
    }

    // Not a web-search request — forward the buffered body unchanged.
    debug!("Not a web-search-only request, forwarding buffered body to upstream");
    let http_client = state.http_client().await;
    let upstream_request = http_client
        .request(method.clone(), upstream_url)
        .headers(upstream_headers)
        .body(body_bytes.to_vec());

    relay_upstream(
        state,
        &method,
        request_target,
        http_client,
        upstream_request,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use http_body_util::BodyExt;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    async fn spawn_upstream_500() -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .fallback(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "upstream internal error") });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        (addr, handle)
    }

    #[tokio::test]
    async fn copilot_stream_keeps_one_message_identity() {
        let source = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"first\",\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"second\",\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"third\",\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"id\":\"fourth\",\"type\":\"message\"}]}}\n\n",
        );
        let app = Router::new().fallback(move || async move {
            ([(header::CONTENT_TYPE, "text/event-stream")], source)
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = ProxyState {
            upstream_url: format!("http://{addr}"),
            http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
            upstream_headers: vec![],
            web_search: None,
            copilot_responses: true,
        };
        let request = Request::builder()
            .method("POST")
            .uri("/responses")
            .body(Body::from("{\"stream\":true}"))
            .unwrap();
        let response = proxy_handler(State(state), request).await.into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        server.abort();
        assert_eq!(
            body.as_ref(),
            source
                .replace("second", "first")
                .replace("third", "first")
                .replace("fourth", "first")
                .as_bytes()
        );
    }

    #[tokio::test]
    async fn normalizer_scope_headers_and_request_bytes_are_preserved() {
        let cases = [
            (
                true,
                "POST",
                "/responses?test=1",
                "text/event-stream; charset=utf-8",
                200,
                None,
                true,
            ),
            (
                true,
                "POST",
                "/v1/responses",
                "TEXT/EVENT-STREAM",
                200,
                Some("identity"),
                true,
            ),
            (
                false,
                "POST",
                "/responses",
                "text/event-stream",
                200,
                None,
                false,
            ),
            (
                true,
                "GET",
                "/responses",
                "text/event-stream",
                200,
                None,
                false,
            ),
            (
                true,
                "POST",
                "/v1/messages",
                "text/event-stream",
                200,
                None,
                false,
            ),
            (
                true,
                "POST",
                "/responses",
                "application/json",
                200,
                None,
                false,
            ),
            (
                true,
                "POST",
                "/responses",
                "text/event-stream",
                400,
                None,
                false,
            ),
            (
                true,
                "POST",
                "/responses",
                "text/event-stream",
                200,
                Some("gzip"),
                false,
            ),
        ];
        for (copilot, method, path, content_type, status, encoding, changed) in cases {
            let source = concat!(
                "data: {\"type\":\"response.future\",\"output_index\":0,\"item_id\":\"first\"}\n\n",
                "data: {\"type\":\"response.future\",\"output_index\":0,\"item_id\":\"later\"}\n\n",
            );
            let request_body = if method == "GET" {
                ""
            } else {
                " {\"future\":1.2300e+04} "
            };
            let identity = copilot && is_responses_request(&method.parse().unwrap(), path);
            let app = Router::new().fallback(move |request: Request<Body>| async move {
                assert_eq!(
                    request.headers()[header::ACCEPT_ENCODING],
                    if identity { "identity" } else { "gzip" }
                );
                assert_eq!(
                    request.into_body().collect().await.unwrap().to_bytes(),
                    request_body
                );
                let mut response = Response::builder()
                    .status(status)
                    .header(header::CONTENT_TYPE, content_type)
                    .header(header::CONTENT_LENGTH, source.len())
                    .header(header::ETAG, "original")
                    .header("x-request-id", "keep");
                if let Some(encoding) = encoding {
                    response = response.header(header::CONTENT_ENCODING, encoding);
                }
                response.body(Body::from(source)).unwrap()
            });
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let state = ProxyState {
                upstream_url: format!("http://{addr}"),
                http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
                upstream_headers: vec![],
                web_search: None,
                copilot_responses: copilot,
            };
            let request = Request::builder()
                .method(method)
                .uri(path)
                .header(header::ACCEPT_ENCODING, "gzip")
                .body(Body::from(request_body))
                .unwrap();
            let response = proxy_handler(State(state), request).await.into_response();
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(response.headers()["x-request-id"], "keep");
            assert_eq!(
                response.headers().contains_key(header::CONTENT_LENGTH),
                !changed
            );
            assert_eq!(response.headers().contains_key(header::ETAG), !changed);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(
                body.as_ref(),
                if changed {
                    source.replace("later", "first")
                } else {
                    source.to_owned()
                }
                .as_bytes()
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn dropping_downstream_cancels_a_stalled_upstream() {
        struct StalledBody {
            first: Option<Bytes>,
            dropped: Arc<tokio::sync::Notify>,
        }
        impl futures::Stream for StalledBody {
            type Item = Result<Bytes, std::io::Error>;
            fn poll_next(
                mut self: std::pin::Pin<&mut Self>,
                _: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                match self.first.take() {
                    Some(bytes) => std::task::Poll::Ready(Some(Ok(bytes))),
                    None => std::task::Poll::Pending,
                }
            }
        }
        impl Drop for StalledBody {
            fn drop(&mut self) {
                self.dropped.notify_one();
            }
        }
        let dropped = Arc::new(tokio::sync::Notify::new());
        let upstream_dropped = dropped.clone();
        let app = Router::new().fallback(move || {
            let dropped = upstream_dropped.clone();
            async move {
                Response::builder()
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from_stream(StalledBody {
                        first: Some(Bytes::from_static(b": ready\n\n")),
                        dropped,
                    }))
                    .unwrap()
            }
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = ProxyState {
            upstream_url: format!("http://{addr}"),
            http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
            upstream_headers: vec![],
            web_search: None,
            copilot_responses: true,
        };
        let request = Request::builder()
            .method("POST")
            .uri("/responses")
            .body(Body::empty())
            .unwrap();
        let response = proxy_handler(State(state), request).await.into_response();
        let mut body = response.into_body();
        let frame = tokio::time::timeout(Duration::from_secs(2), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(frame.into_data().unwrap(), b": ready\n\n".as_slice());
        drop(body);
        tokio::time::timeout(Duration::from_secs(2), dropped.notified())
            .await
            .unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn upstream_stream_error_preserves_partial_event_and_reports_failure() {
        let app = Router::new().fallback(|| async {
            let stream = futures::stream::unfold(0, |step| async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let next = match step {
                    0 => Ok(Bytes::from_static(b": ready\n\n")),
                    1 => Ok(Bytes::from_static(b"data: {\"type\":")),
                    2 => Err(std::io::Error::other("upstream disconnected")),
                    _ => return None,
                };
                Some((next, step + 1))
            });
            Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let state = ProxyState {
            upstream_url: format!("http://{addr}"),
            http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
            upstream_headers: vec![],
            web_search: None,
            copilot_responses: true,
        };
        let request = Request::builder()
            .method("POST")
            .uri("/responses")
            .body(Body::empty())
            .unwrap();
        let response = proxy_handler(State(state), request).await.into_response();
        let mut body = response.into_body();
        let mut bytes = Vec::new();
        let mut failed = false;
        while let Some(frame) = tokio::time::timeout(Duration::from_secs(2), body.frame())
            .await
            .unwrap()
        {
            match frame {
                Ok(frame) => bytes.extend_from_slice(&frame.into_data().unwrap()),
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        assert!(failed);
        assert_eq!(bytes, b": ready\n\ndata: {\"type\":");
        server.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn upstream_500_is_delayed_then_converted_to_503() {
        let (addr, upstream_server) = spawn_upstream_500().await;
        let state = ProxyState {
            upstream_url: format!("http://{}", addr),
            http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
            upstream_headers: vec![],
            web_search: None,
            copilot_responses: false,
        };
        let request = Request::builder()
            .uri("/v1/messages")
            .body(Body::empty())
            .unwrap();

        let response_task =
            tokio::spawn(async move { proxy_handler(State(state), request).await.into_response() });

        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(UPSTREAM_500_RESPONSE_DELAY_SECS - 1)).await;
        tokio::task::yield_now().await;
        assert!(!response_task.is_finished());

        tokio::time::advance(Duration::from_secs(1)).await;
        let mut response = response_task.await.unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
        assert_eq!(
            response.headers().get(header::CONNECTION),
            Some(&HeaderValue::from_static("close"))
        );

        let body = response.body_mut().collect().await.unwrap().to_bytes();
        assert!(body.is_empty());

        upstream_server.abort();
    }
}
