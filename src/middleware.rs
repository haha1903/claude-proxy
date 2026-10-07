use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Method, Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock as SyncRwLock},
};
use tracing::{debug, warn};

use crate::auth::{create_upstream_auth, CopilotAuth, UpstreamAuth};
use crate::config::ProxyConfig;

/// State for API key validation middleware
#[derive(Clone)]
pub struct ApiKeyValidatorState {
    clients: Arc<SyncRwLock<HashMap<String, Route>>>,
    tokens: Arc<SyncRwLock<HashMap<String, Arc<dyn UpstreamAuth>>>>,
    allow_missing_key: bool,
}

#[derive(Clone)]
struct Route {
    name: String,
    members: Vec<(String, Arc<dyn UpstreamAuth>)>,
}

impl Route {
    fn single(auth: Arc<dyn UpstreamAuth>) -> Self {
        Self {
            name: String::new(),
            members: vec![(String::new(), auth)],
        }
    }

    fn select(
        &self,
        headers: &HeaderMap,
        method: &Method,
        path: &str,
    ) -> Result<Arc<dyn UpstreamAuth>, StatusCode> {
        if self.members.len() == 1 {
            return Ok(self.members[0].1.clone());
        }
        let session = session_id(headers)?;
        let discovery = method == Method::GET && matches!(path, "/models" | "/v1/models");
        let session = session
            .or(if discovery {
                Some("model-discovery")
            } else {
                None
            })
            .ok_or(StatusCode::BAD_REQUEST)?;
        self.members
            .iter()
            .max_by_key(|(login, _)| {
                (
                    crate::routing::rendezvous_score(&self.name, session, login),
                    login,
                )
            })
            .map(|(_, auth)| auth.clone())
            .ok_or(StatusCode::UNAUTHORIZED)
    }
}

fn session_id(headers: &HeaderMap) -> Result<Option<&str>, StatusCode> {
    let mut session = None;
    for name in [
        "session-id",
        "session_id",
        "thread-id",
        "x-claude-code-session-id",
    ] {
        for value in headers.get_all(name) {
            let value = value.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
            if value.is_empty()
                || value.len() > 256
                || !value.bytes().all(|b| b.is_ascii_graphic())
                || session.is_some_and(|previous| previous != value)
            {
                return Err(StatusCode::BAD_REQUEST);
            }
            session = Some(value);
        }
    }
    Ok(session)
}

impl ApiKeyValidatorState {
    pub fn validate_config(config: &ProxyConfig) -> Result<(), &'static str> {
        if let Some(url) = &config.copilot_vault_url {
            if config.copilot_routing.is_some() {
                return Err("Configure only one Copilot routing source");
            }
            crate::keyvault::validate_vault_url(url)?;
            crate::routing::validate_headers(&config.upstream_headers)
        } else if let Some(routing) = &config.copilot_routing {
            routing.validate(&config.upstream_headers)
        } else if config.client_api_key.trim().is_empty() || config.upstream_auth.is_none() {
            Err("client_api_key and upstream_auth are required without copilot_routing")
        } else {
            Ok(())
        }
    }

    pub fn from_config(config: &ProxyConfig) -> Result<Self, &'static str> {
        Self::validate_config(config)?;
        let state = Self {
            clients: Arc::new(SyncRwLock::new(HashMap::new())),
            tokens: Arc::new(SyncRwLock::new(HashMap::new())),
            allow_missing_key: config.copilot_routing.is_none()
                && config.copilot_vault_url.is_none(),
        };
        if let Some(routing) = &config.copilot_routing {
            state.replace_copilot(routing)?;
        } else if config.copilot_vault_url.is_none() {
            let auth = create_upstream_auth(config.upstream_auth.as_ref().unwrap());
            state
                .clients
                .write()
                .map_err(|_| "Routing lock failed")?
                .insert(config.client_api_key.clone(), Route::single(auth));
        }
        Ok(state)
    }

    #[cfg(test)]
    pub fn lookup(&self, key: &str) -> Option<Arc<dyn UpstreamAuth>> {
        self.clients
            .read()
            .ok()?
            .get(key)?
            .members
            .first()
            .map(|(_, auth)| auth.clone())
    }

    pub fn replace_records(
        &self,
        records: &HashMap<String, crate::routing::VaultRecord>,
    ) -> Result<(), &'static str> {
        let mut previous = self.tokens.write().map_err(|_| "Routing lock failed")?;
        let mut tokens = HashMap::new();
        let mut clients = HashMap::new();
        for (name, record) in records {
            if !record.validate() || clients.contains_key(&record.api_key) {
                return Err("Invalid Vault routing");
            }
            let mut members = Vec::new();
            for account in &record.github {
                let auth = tokens.entry(account.token.clone()).or_insert_with(|| {
                    previous
                        .get(&account.token)
                        .cloned()
                        .unwrap_or_else(|| Arc::new(CopilotAuth::new(account.token.clone())))
                });
                members.push((account.login.to_ascii_lowercase(), auth.clone()));
            }
            clients.insert(
                record.api_key.clone(),
                Route {
                    name: name.clone(),
                    members,
                },
            );
        }
        *self.clients.write().map_err(|_| "Routing lock failed")? = clients;
        *previous = tokens;
        Ok(())
    }

    pub fn replace_copilot(
        &self,
        routing: &crate::routing::CopilotRouting,
    ) -> Result<(), &'static str> {
        if !routing.accounts.is_empty() || !routing.clients.is_empty() {
            routing.validate(&[])?;
        }
        let mut previous = self.tokens.write().map_err(|_| "Routing lock failed")?;
        let mut tokens = HashMap::new();
        let mut accounts = HashMap::new();
        for account in &routing.accounts {
            let auth = tokens
                .entry(account.github_token.clone())
                .or_insert_with(|| {
                    previous
                        .get(&account.github_token)
                        .cloned()
                        .unwrap_or_else(|| Arc::new(CopilotAuth::new(account.github_token.clone())))
                });
            accounts.insert(&account.name, auth.clone());
        }
        let clients = routing
            .clients
            .iter()
            .map(|client| {
                (
                    client.api_key.clone(),
                    Route::single(accounts[&client.account].clone()),
                )
            })
            .collect();
        // Publish the complete map once. In-flight requests retain their selected Arc.
        *self.clients.write().map_err(|_| "Routing lock failed")? = clients;
        *previous = tokens;
        Ok(())
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

    let route = api_key
        .as_deref()
        .and_then(|key| state.clients.read().ok()?.get(key).cloned());
    let selected = route
        .map(|route| route.select(request.headers(), request.method(), request.uri().path()))
        .transpose()?;
    match (api_key, selected) {
        (Some(_), Some(auth)) => {
            request.extensions_mut().insert(SelectedUpstreamAuth(auth));
            Ok(next.run(request).await)
        }
        (Some(_), None) => {
            warn!("Invalid API key provided");
            Err(StatusCode::UNAUTHORIZED)
        }
        (None, _) if state.allow_missing_key => {
            // No API key provided - relay without upstream auth
            debug!("No API key provided, relaying without upstream authentication");
            Ok(next.run(request).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
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
            &state.lookup("client-one").unwrap(),
            &state.lookup("client-shared").unwrap()
        ));
        assert!(!Arc::ptr_eq(
            &state.lookup("client-one").unwrap(),
            &state.lookup("client-two").unwrap()
        ));
        config.copilot_routing.as_mut().unwrap().accounts[1].github_token = "github-one".into();
        let state = ApiKeyValidatorState::from_config(&config).unwrap();
        assert!(Arc::ptr_eq(
            &state.lookup("client-one").unwrap(),
            &state.lookup("client-two").unwrap()
        ));
        config.copilot_routing.as_mut().unwrap().clients[0].account = "unknown".into();
        assert!(ApiKeyValidatorState::from_config(&config).is_err());
    }

    fn state(allow_missing_key: bool) -> ApiKeyValidatorState {
        let one: Arc<dyn UpstreamAuth> = Arc::new(BearerAuth::new("upstream-one".into()));
        let two: Arc<dyn UpstreamAuth> = Arc::new(BearerAuth::new("upstream-two".into()));
        ApiKeyValidatorState {
            clients: Arc::new(SyncRwLock::new(HashMap::from([
                ("client-one".into(), Route::single(one.clone())),
                ("client-shared".into(), Route::single(one)),
                ("client-two".into(), Route::single(two)),
            ]))),
            tokens: Arc::new(SyncRwLock::new(HashMap::new())),
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
        let state = state(false);
        state
            .clients
            .write()
            .unwrap()
            .insert("client-one".into(), Route::single(Arc::new(FailedAuth)));
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
    fn pool(logins: &[&str]) -> Route {
        Route {
            name: "copilot-5".into(),
            members: logins
                .iter()
                .map(|s| {
                    (
                        (*s).into(),
                        Arc::new(BearerAuth::new((*s).into())) as Arc<dyn UpstreamAuth>,
                    )
                })
                .collect(),
        }
    }
    async fn selected(route: &Route, session: &str) -> String {
        let mut headers = HeaderMap::new();
        headers.insert("session-id", session.parse().unwrap());
        route
            .select(&headers, &Method::POST, "/responses")
            .unwrap()
            .get_auth_header()
            .await
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned()
    }

    #[tokio::test]
    async fn rendezvous_survives_reorder_and_changes_only_required_sessions() {
        let original = pool(&["alice", "bob", "charlie"]);
        let reordered = pool(&["charlie", "alice", "bob"]);
        let added = pool(&["alice", "bob", "charlie", "dave"]);
        let removed = pool(&["alice", "charlie"]);
        let mut moved = 0;
        for i in 0..1000 {
            let session = format!("session-{i}");
            let first = selected(&original, &session).await;
            assert_eq!(first, selected(&reordered, &session).await);
            let next = selected(&added, &session).await;
            if next != first {
                assert_eq!(next, "Bearer dave");
                moved += 1;
            }
            if first != "Bearer bob" {
                assert_eq!(first, selected(&removed, &session).await);
            }
        }
        assert!((180..320).contains(&moved));
        let mut changed = original.clone();
        for (_, auth) in &mut changed.members {
            *auth = Arc::new(BearerAuth::new("rotated".into()));
        }
        let scores = |route: &Route| {
            route
                .members
                .iter()
                .map(|(login, _)| crate::routing::rendezvous_score(&route.name, "stable", login))
                .collect::<Vec<_>>()
        };
        assert_eq!(scores(&original), scores(&changed));
        assert_eq!(
            crate::routing::rendezvous_score("pool", "session", "Alice"),
            crate::routing::rendezvous_score("pool", "session", "alice")
        );
    }

    #[test]
    fn multiple_accounts_require_unambiguous_stable_session_headers() {
        let route = pool(&["alice", "bob"]);
        let mut headers = HeaderMap::new();
        assert!(route.select(&headers, &Method::POST, "/responses").is_err());
        assert!(route.select(&headers, &Method::GET, "/models").is_ok());
        assert!(route.select(&headers, &Method::GET, "/v1/models").is_ok());
        for name in [
            "session-id",
            "session_id",
            "thread-id",
            "x-claude-code-session-id",
        ] {
            headers.insert(name, "stable".parse().unwrap());
            assert_eq!(session_id(&headers).unwrap(), Some("stable"));
        }
        headers.insert("thread-id", "different".parse().unwrap());
        assert!(route.select(&headers, &Method::POST, "/responses").is_err());
        headers.clear();
        headers.insert("session-id", "".parse().unwrap());
        assert!(session_id(&headers).is_err());
        headers.insert("session-id", "a".repeat(257).parse().unwrap());
        assert!(session_id(&headers).is_err());
        headers.clear();
        headers.insert("x-client-request-id", "request-only".parse().unwrap());
        assert!(route.select(&headers, &Method::POST, "/messages").is_err());
        assert!(pool(&["alice"])
            .select(&headers, &Method::POST, "/messages")
            .is_ok());
    }

    #[test]
    fn invalid_pool_refresh_is_atomic() {
        let state = state(false);
        let mut records = HashMap::new();
        let good: crate::routing::VaultRecord = serde_json::from_value(serde_json::json!({"api_key":"new-key","github":[{"login":"alice","token":"fixture-token"},{"login":"bob","token":"fixture-bob"}]})).unwrap();
        records.insert("copilot-5".into(), good.clone());
        state.replace_records(&records).unwrap();
        let before = state.lookup("new-key").unwrap();
        records.get_mut("copilot-5").unwrap().github[0].token = "rotated-token".into();
        state.replace_records(&records).unwrap();
        assert!(!Arc::ptr_eq(&before, &state.lookup("new-key").unwrap()));
        records.insert("copilot-6".into(), good.clone());
        assert!(state.replace_records(&records).is_err());
        assert!(state.lookup("new-key").is_some());
        for mutate in [
            |r: &mut crate::routing::VaultRecord| r.policy = "most_remaining".into(),
            |r: &mut crate::routing::VaultRecord| r.github.clear(),
            |r: &mut crate::routing::VaultRecord| r.github[1].login = "ALICE".into(),
            |r: &mut crate::routing::VaultRecord| r.github[0].token = "bad token".into(),
        ] {
            let mut record = good.clone();
            mutate(&mut record);
            assert!(!record.validate());
        }
        state.replace_records(&HashMap::new()).unwrap();
        assert!(state.lookup("new-key").is_none());
    }
}
