use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, warn};

use crate::auth::{create_upstream_auth, CopilotAuth, UpstreamAuth};
use crate::config::ProxyConfig;

/// State for API key validation middleware
#[derive(Clone)]
pub struct ApiKeyValidatorState {
    clients: HashMap<String, Arc<dyn UpstreamAuth>>,
    allow_missing_key: bool,
}

impl ApiKeyValidatorState {
    pub fn validate_config(config: &ProxyConfig) -> Result<(), &'static str> {
        if let Some(routing) = &config.copilot_routing {
            routing.validate(&config.upstream_headers)
        } else if config.client_api_key.trim().is_empty() || config.upstream_auth.is_none() {
            Err("client_api_key and upstream_auth are required without copilot_routing")
        } else {
            Ok(())
        }
    }

    pub fn from_config(config: &ProxyConfig) -> Result<Self, &'static str> {
        Self::validate_config(config)?;
        if let Some(routing) = &config.copilot_routing {
            let mut tokens: HashMap<&str, Arc<dyn UpstreamAuth>> = HashMap::new();
            let mut accounts = HashMap::new();
            for account in &routing.accounts {
                // Repeated GitHub tokens share one cache, even under different names.
                let auth = tokens
                    .entry(&account.github_token)
                    .or_insert_with(|| Arc::new(CopilotAuth::new(account.github_token.clone())));
                accounts.insert(&account.name, auth.clone());
            }
            let clients = routing
                .clients
                .iter()
                .map(|client| (client.api_key.clone(), accounts[&client.account].clone()))
                .collect();
            Ok(Self {
                clients,
                allow_missing_key: false,
            })
        } else {
            let auth = create_upstream_auth(config.upstream_auth.as_ref().unwrap());
            Ok(Self {
                clients: HashMap::from([(config.client_api_key.clone(), auth)]),
                allow_missing_key: true,
            })
        }
    }
}

/// The selected provider belongs to this request and never changes on failure.
#[derive(Clone)]
pub struct SelectedUpstreamAuth(pub Arc<dyn UpstreamAuth>);

/// Middleware to validate client API key
pub async fn validate_client_api_key(
    State(state): State<ApiKeyValidatorState>,
    mut request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract API key from request headers
    // Support both api-key / x-api-key header and Authorization: Bearer token
    let api_key = request
        .headers()
        .get("api-key")
        .or(request.headers().get("x-api-key"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            request
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        });

    match api_key {
        Some(key) if state.clients.contains_key(&key) => {
            request
                .extensions_mut()
                .insert(SelectedUpstreamAuth(state.clients[&key].clone()));
            Ok(next.run(request).await)
        }
        Some(_) => {
            warn!("Invalid API key provided");
            Err(StatusCode::UNAUTHORIZED)
        }
        None if state.allow_missing_key => {
            // No API key provided - relay without upstream auth
            debug!("No API key provided, relaying without upstream authentication");
            Ok(next.run(request).await)
        }
        None => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AuthError, BearerAuth};
    use crate::proxy::{build_http_client, proxy_handler, ProxyState};
    use crate::routing::tests::routing;
    use axum::{routing::any, Router};
    use http_body_util::BodyExt;
    use reqwest::header::HeaderValue;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    fn config() -> ProxyConfig {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }

    #[test]
    fn configuration_shares_repeated_tokens_and_rejects_invalid_bindings() {
        let mut config = config();
        assert!(ApiKeyValidatorState::from_config(&config).is_err());
        config.client_api_key = "legacy".into();
        config.upstream_auth = Some(crate::config::UpstreamAuthConfig::Bearer {
            token: "upstream".into(),
        });
        assert!(
            ApiKeyValidatorState::from_config(&config)
                .unwrap()
                .allow_missing_key
        );
        config.copilot_routing = Some(routing());
        let state = ApiKeyValidatorState::from_config(&config).unwrap();
        assert!(!state.allow_missing_key);
        assert!(Arc::ptr_eq(
            &state.clients["client-one"],
            &state.clients["client-shared"]
        ));
        assert!(!Arc::ptr_eq(
            &state.clients["client-one"],
            &state.clients["client-two"]
        ));
        config.copilot_routing.as_mut().unwrap().accounts[1].github_token = "github-one".into();
        let state = ApiKeyValidatorState::from_config(&config).unwrap();
        assert!(Arc::ptr_eq(
            &state.clients["client-one"],
            &state.clients["client-two"]
        ));
        config.copilot_routing.as_mut().unwrap().clients[0].account = "unknown".into();
        assert!(ApiKeyValidatorState::from_config(&config).is_err());
    }

    fn state(allow_missing_key: bool) -> ApiKeyValidatorState {
        let one: Arc<dyn UpstreamAuth> = Arc::new(BearerAuth::new("upstream-one".into()));
        let two: Arc<dyn UpstreamAuth> = Arc::new(BearerAuth::new("upstream-two".into()));
        ApiKeyValidatorState {
            clients: HashMap::from([
                ("client-one".into(), one.clone()),
                ("client-shared".into(), one),
                ("client-two".into(), two),
            ]),
            allow_missing_key,
        }
    }

    fn app(state: ApiKeyValidatorState, upstream_url: String) -> Router {
        Router::new()
            .fallback(any(proxy_handler))
            .layer(axum::middleware::from_fn_with_state(
                state,
                validate_client_api_key,
            ))
            .with_state(ProxyState {
                upstream_url,
                http_client: Arc::new(RwLock::new(build_http_client().unwrap())),
                upstream_headers: vec![],
                web_search: None,
                copilot_responses: true,
            })
    }

    async fn upstream() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let app = Router::new().fallback(move |request: Request<Body>| {
            count.fetch_add(1, Ordering::SeqCst);
            async move {
                assert!(!request.headers().contains_key("api-key"));
                assert!(!request.headers().contains_key("x-api-key"));
                let auth = request
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("none")
                    .to_owned();
                if request.uri().path() == "/denied" {
                    (StatusCode::FORBIDDEN, auth)
                } else {
                    (StatusCode::OK, auth)
                }
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, calls, server)
    }

    #[tokio::test]
    async fn concurrent_requests_keep_their_account_and_strip_client_credentials() {
        let (url, calls, server) = upstream().await;
        let app = app(state(false), url);
        let mut tasks = Vec::new();
        for i in 0..30 {
            let app = app.clone();
            tasks.push(tokio::spawn(async move {
                let (key, expected) = match i % 3 {
                    0 => ("client-one", "Bearer upstream-one"),
                    1 => ("client-two", "Bearer upstream-two"),
                    _ => ("client-shared", "Bearer upstream-one"),
                };
                let header = ["api-key", "x-api-key", "authorization"][i % 3];
                let value = if header == "authorization" {
                    format!("Bearer {key}")
                } else {
                    key.into()
                };
                let response = app
                    .oneshot(
                        Request::builder()
                            .uri("/models")
                            .header(header, value)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(
                    response.into_body().collect().await.unwrap().to_bytes(),
                    expected
                );
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 30);
        server.abort();
    }

    #[tokio::test]
    async fn missing_unknown_and_conflicting_keys_never_reach_upstream() {
        let (url, calls, server) = upstream().await;
        let app = app(state(false), url);
        for headers in [
            vec![],
            vec![("authorization", "Bearer wrong")],
            vec![("authorization", "Basic wrong")],
            vec![("api-key", "wrong"), ("authorization", "Bearer client-one")],
        ] {
            let mut request = Request::builder().uri("/models");
            for (name, value) in headers {
                request = request.header(name, value);
            }
            let response = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        server.abort();
    }

    #[tokio::test]
    async fn legacy_missing_key_is_passthrough_and_upstream_failure_never_switches_accounts() {
        let (url, calls, server) = upstream().await;
        let legacy = app(state(true), url.clone());
        let response = legacy
            .oneshot(Request::builder().body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            "none"
        );
        let response = app(state(false), url)
            .oneshot(
                Request::builder()
                    .uri("/denied")
                    .header("api-key", "client-one")
                    .header("authorization", "Bearer client-two")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            "Bearer upstream-one"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        server.abort();
    }

    struct FailedAuth;
    #[async_trait::async_trait]
    impl UpstreamAuth for FailedAuth {
        async fn get_auth_header(&self) -> Result<HeaderValue, AuthError> {
            Err(AuthError::TokenAcquisition("denied".into()))
        }
    }

    #[tokio::test]
    async fn token_exchange_failure_does_not_use_another_account() {
        let (url, calls, server) = upstream().await;
        let mut state = state(false);
        state
            .clients
            .insert("client-one".into(), Arc::new(FailedAuth));
        let response = app(state, url)
            .oneshot(
                Request::builder()
                    .header("api-key", "client-one")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        server.abort();
    }
}
