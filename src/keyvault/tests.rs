use super::*;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Router,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Fixture {
    origin: String,
    pages: Mutex<HashMap<String, (StatusCode, Value)>>,
    calls: Mutex<Vec<String>>,
}

async fn fixture() -> (Vault, Arc<Fixture>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let fixture = Arc::new(Fixture {
        origin: origin.clone(),
        pages: Mutex::new(HashMap::new()),
        calls: Mutex::new(vec![]),
    });
    let f = fixture.clone();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let f = f.clone();
        async move {
            let path = request.uri().path();
            f.calls.lock().unwrap().push(request.uri().to_string());
            if path == "/identity" {
                assert_eq!(request.headers()["x-identity-header"], "identity-fixture");
            } else if path.starts_with("/secrets") {
                assert_eq!(request.headers()["authorization"], "Bearer vault-fixture");
            } else {
                assert!(request.headers()["authorization"]
                    .to_str()
                    .unwrap()
                    .starts_with("token "));
            }
            let pages = f.pages.lock().unwrap();
            let (status, value) = pages
                .get(&request.uri().to_string())
                .or_else(|| pages.get(path))
                .cloned()
                .unwrap_or((StatusCode::NOT_FOUND, json!({})));
            (status, axum::Json(value)).into_response()
        }
    });
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let vault = Vault {
        url: Url::parse(&(origin.clone() + "/")).unwrap(),
        http: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap(),
        identity_endpoint: origin.clone() + "/identity",
        identity_header: "identity-fixture".into(),
        github: origin,
        records: HashMap::new(),
    };
    fixture.set("/identity", json!({"access_token":"vault-fixture"}));
    fixture.list(&["copilot-1"]);
    fixture.record("copilot-1", record("key-one", "alice", "token-one"));
    fixture.set("/copilot_internal/user", json!({"login":"alice"}));
    (vault, fixture, server)
}

impl Fixture {
    fn set(&self, path: &str, value: Value) {
        self.pages
            .lock()
            .unwrap()
            .insert(path.into(), (StatusCode::OK, value));
    }
    fn list(&self, names: &[&str]) {
        self.set("/secrets", json!({"value": names.iter().map(|name| json!({"id":format!("{}/secrets/{name}", self.origin), "attributes":{"enabled":true}})).collect::<Vec<_>>() }));
    }
    fn record(&self, name: &str, value: Value) {
        self.set(
            &format!("/secrets/{name}"),
            json!({"value":value.to_string(),"attributes":{"enabled":true}}),
        );
    }
}
fn record(key: &str, login: &str, token: &str) -> Value {
    json!({"api_key":key,"github":[{"login":login,"token":token}]})
}
fn state() -> ApiKeyValidatorState {
    let config =
        serde_json::from_value(json!({"copilot_vault_url":"https://fixture.vault.azure.net/"}))
            .unwrap();
    ApiKeyValidatorState::from_config(&config).unwrap()
}

#[test]
fn validates_origins_names_and_activation() {
    assert!(validate_vault_url("https://fixture.vault.azure.net/").is_ok());
    for url in [
        "bad",
        "http://fixture.vault.azure.net",
        "https://vault.azure.net",
        "https://a.b.vault.azure.net",
        "https://x.vault.azure.net:444",
        "https://user@x.vault.azure.net",
        "https://x.vault.azure.net/path",
        "https://x.vault.azure.net?x=1",
        "https://x.vault.azure.net#x",
    ] {
        assert!(validate_vault_url(url).is_err(), "{url}");
    }
    assert_eq!(number("copilot-1"), Some(1));
    for name in [
        "copilot-0",
        "copilot-01",
        "copilot-x",
        "copilot-",
        "copilot-4294967296",
        "copilot-account-1",
        "req-1",
    ] {
        assert_eq!(number(name), None);
    }
    assert!(enabled(&json!({}), 10));
    assert!(enabled(&json!({"nbf":10,"exp":11}), 10));
    for value in [
        json!({"enabled":false}),
        json!({"nbf":11}),
        json!({"exp":10}),
    ] {
        assert!(!enabled(&value, 10));
    }
    assert!(!credential("bad token"));
    assert!(!credential(""));
}

#[tokio::test]
async fn runtime_reuses_auth_rotates_keys_and_removes_deleted_records() {
    let (mut vault, f, server) = fixture().await;
    let state = state();
    vault.refresh(&state).await.unwrap();
    let before = state.lookup("key-one").unwrap();
    let initial_calls = f.calls.lock().unwrap().len();
    vault.refresh(&state).await.unwrap();
    assert_eq!(f.calls.lock().unwrap().len() - initial_calls, 3);
    assert!(Arc::ptr_eq(&before, &state.lookup("key-one").unwrap()));
    f.record("copilot-1", record("key-two", "Alice", "token-one"));
    vault.refresh(&state).await.unwrap();
    assert!(state.lookup("key-one").is_none());
    assert!(Arc::ptr_eq(&before, &state.lookup("key-two").unwrap()));
    f.record("copilot-1", record("key-two", "alice", "rotated-token"));
    vault.refresh(&state).await.unwrap();
    assert!(!Arc::ptr_eq(&before, &state.lookup("key-two").unwrap()));
    f.list(&[]);
    vault.refresh(&state).await.unwrap();
    assert!(state.lookup("key-two").is_none());
    // The old selected provider remains alive for its already-started request.
    assert_eq!(Arc::strong_count(&before), 1);
    server.abort();
}

#[tokio::test]
async fn partial_transient_failure_retains_verified_value_and_bad_updates_disable() {
    let (mut vault, f, server) = fixture().await;
    let state = state();
    vault.refresh(&state).await.unwrap();
    f.pages.lock().unwrap().insert(
        "/secrets/copilot-1".into(),
        (StatusCode::SERVICE_UNAVAILABLE, json!({})),
    );
    vault.refresh(&state).await.unwrap();
    assert!(state.lookup("key-one").is_some());
    for value in [
        json!({"value":"bad"}),
        json!({"value":record("bad key","alice","token").to_string()}),
        json!({"attributes":{"enabled":false}}),
    ] {
        f.set("/secrets/copilot-1", value);
        vault.refresh(&state).await.unwrap();
        assert!(state.lookup("key-one").is_none());
    }
    f.record("copilot-1", record("new-key", "bob", "bob-token"));
    vault.refresh(&state).await.unwrap();
    assert!(state.lookup("new-key").is_none());
    f.set("/copilot_internal/user", json!({}));
    f.set("/user", json!({"login":"bob"}));
    vault.refresh(&state).await.unwrap();
    assert!(state.lookup("new-key").is_some());
    f.set("/identity", json!({}));
    assert!(vault.refresh(&state).await.is_err());
    assert!(state.lookup("new-key").is_some());
    server.abort();
}

#[tokio::test]
async fn pools_allow_overlapping_accounts_but_not_duplicate_client_keys() {
    let (mut vault, f, server) = fixture().await;
    let state = state();
    f.list(&["copilot-1", "copilot-2", "req-unrelated"]);
    f.record("copilot-2", record("key-two", "alice", "token-one"));
    vault.refresh(&state).await.unwrap();
    assert_eq!(vault.records.len(), 2);
    assert!(Arc::ptr_eq(
        &state.lookup("key-one").unwrap(),
        &state.lookup("key-two").unwrap()
    ));
    assert!(!f
        .calls
        .lock()
        .unwrap()
        .iter()
        .any(|s| s.contains("req-unrelated")));
    f.record("copilot-2", record("key-one", "alice", "token-one"));
    vault.refresh(&state).await.unwrap();
    assert!(vault.records.is_empty());
    server.abort();
}

#[tokio::test]
async fn pagination_checks_every_origin_and_detects_cycles() {
    let (vault, f, server) = fixture().await;
    let first = json!({"value":[],"nextLink":format!("{}/secrets?page=2",f.origin)});
    f.set("/secrets?api-version=7.4", first.clone());
    f.set(
        "/secrets?page=2",
        json!({"value":[{"id":format!("{}/secrets/copilot-2",f.origin)}]}),
    );
    assert_eq!(
        vault.list("vault-fixture").await.unwrap(),
        vec!["copilot-2"]
    );
    for page in [
        json!({"value":[],"nextLink":"https://evil.invalid/secrets"}),
        json!({"value":[],"nextLink":false}),
        json!({"value":[] ,"nextLink":"bad"}),
        json!({"value":[{"id":"https://evil.invalid/secrets/copilot-1"}]}),
        json!({"value":[{"id":format!("{}/wrong",f.origin)}]}),
        json!({"value":false}),
    ] {
        f.set("/secrets?api-version=7.4", page);
        assert!(vault.list("vault-fixture").await.is_err());
    }
    f.set("/secrets?api-version=7.4", first.clone());
    f.set("/secrets?page=2", first);
    assert!(vault.list("vault-fixture").await.is_err());
    server.abort();
}

#[tokio::test]
async fn transport_rejects_redirects_invalid_json_and_oversized_bodies() {
    for (status, body) in [
        (StatusCode::FOUND, "{}".into()),
        (StatusCode::OK, "bad json".into()),
        (StatusCode::OK, "x".repeat(LIMIT + 1)),
    ] {
        let app = Router::new().fallback(move || {
            let body = body.clone();
            async move {
                Response::builder()
                    .status(status)
                    .header("location", "http://127.0.0.1:1/leak")
                    .body(Body::from(body))
                    .unwrap()
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        assert!(json(client.get(url)).await.is_err());
        server.abort();
    }
}
