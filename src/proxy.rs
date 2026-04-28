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

use crate::auth::UpstreamAuth;
use crate::middleware::ClientAuthenticated;

/// Shared state for the proxy
#[derive(Clone)]
pub struct ProxyState {
    pub upstream_url: String,
    pub upstream_auth: Arc<dyn UpstreamAuth>,
    pub http_client: Arc<RwLock<reqwest::Client>>,
    pub upstream_headers: Vec<(String, String)>,
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
        if name_str == "x-api-key" || name_str == "authorization" {
            debug!("  {}: [REDACTED]", name);
        } else {
            debug!("  {}: {:?}", name, value);
        }
    }

    // Build the upstream URL
    let upstream_url = format!("{}{}", state.upstream_url, request_target);

    // Check if the client is authenticated
    let client_authenticated = request
        .extensions()
        .get::<ClientAuthenticated>()
        .map(|auth| auth.0)
        .unwrap_or(false);

    // Build headers for upstream request
    let mut upstream_headers = HeaderMap::new();

    // Only add upstream authentication if the client provided a valid API key
    if client_authenticated {
        // Get auth header for upstream
        let auth_header = match state.upstream_auth.get_auth_header().await {
            Ok(header) => header,
            Err(e) => {
                error!("Failed to get upstream auth header: {}", e);
                return Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!("Authentication error: {}", e)))
                    .unwrap();
            }
        };

        let auth_header_name = state.upstream_auth.auth_header_name();
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
        match state.upstream_auth.get_additional_headers().await {
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

    // Stream all response bodies without buffering
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);

    let mut byte_stream = upstream_response.bytes_stream();
    tokio::spawn(async move {
        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if tx.send(Ok(bytes)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthError;
    use async_trait::async_trait;
    use axum::Router;
    use http_body_util::BodyExt;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    struct UnusedAuth;

    #[async_trait]
    impl UpstreamAuth for UnusedAuth {
        async fn get_auth_header(&self) -> Result<HeaderValue, AuthError> {
            Ok(HeaderValue::from_static("unused"))
        }
    }

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

    #[tokio::test(start_paused = true)]
    async fn upstream_500_is_delayed_then_converted_to_503() {
        let (addr, upstream_server) = spawn_upstream_500().await;
        let state = ProxyState {
            upstream_url: format!("http://{}", addr),
            upstream_auth: Arc::new(UnusedAuth),
            http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
            upstream_headers: vec![],
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
