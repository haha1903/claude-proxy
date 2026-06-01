//! Brave Search API provider.

use async_trait::async_trait;
use serde::Deserialize;
use tracing::debug;

use super::{
    parse_search_response, SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult,
};

const BRAVE_SEARCH_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
/// Brave caps `count` at 20.
const BRAVE_MAX_COUNT: u32 = 20;

/// Web search via the Brave Search API.
pub struct BraveProvider {
    api_key: String,
    endpoint: String,
    http_client: reqwest::Client,
}

impl BraveProvider {
    /// Create a Brave provider using the production endpoint.
    pub fn new(api_key: String, http_client: reqwest::Client) -> Self {
        Self {
            api_key,
            endpoint: BRAVE_SEARCH_ENDPOINT.to_string(),
            http_client,
        }
    }

    /// Create a Brave provider with a custom endpoint (used by tests).
    #[cfg(test)]
    pub fn with_endpoint(api_key: String, endpoint: String, http_client: reqwest::Client) -> Self {
        Self {
            api_key,
            endpoint,
            http_client,
        }
    }
}

#[async_trait]
impl SearchProvider for BraveProvider {
    fn name(&self) -> &'static str {
        "brave"
    }

    async fn search(&self, req: &SearchRequest) -> Result<SearchResponse, SearchError> {
        let count = req.effective_max_results().min(BRAVE_MAX_COUNT);

        debug!("Brave search query: {}", req.query);

        let response = self
            .http_client
            .get(&self.endpoint)
            .query(&[("q", req.query.as_str()), ("count", &count.to_string())])
            .header("Accept", "application/json")
            .header("X-Subscription-Token", &self.api_key)
            .send()
            .await?;

        let raw: BraveResponse = parse_search_response(response).await?;

        let results = raw
            .web
            .map(|w| w.results)
            .unwrap_or_default()
            .into_iter()
            .map(|r| SearchResult {
                url: r.url,
                title: r.title,
                snippet: r.description,
                page_age: r.age.filter(|s| !s.is_empty()),
            })
            .collect();

        Ok(SearchResponse { results })
    }
}

/// Minimal subset of the Brave Search API response.
#[derive(Debug, Deserialize)]
struct BraveResponse {
    web: Option<BraveWeb>,
}

#[derive(Debug, Deserialize)]
struct BraveWeb {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(Debug, Deserialize)]
struct BraveResult {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    age: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Router};
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    /// Spawn a mock Brave endpoint returning the given JSON body, and return its URL.
    async fn spawn_mock(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/res/v1/web/search",
            get(move || async move { ([("content-type", "application/json")], body) }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{}/res/v1/web/search", addr), handle)
    }

    #[tokio::test]
    async fn parses_results_and_page_age() {
        let body = r#"{"web":{"results":[
            {"url":"https://go.dev","title":"Go","description":"Go lang","age":"1 day"},
            {"url":"https://pkg.go.dev","title":"Pkg","description":"Packages"}
        ]}}"#;
        let (endpoint, server) = spawn_mock(body).await;

        let provider =
            BraveProvider::with_endpoint("k".to_string(), endpoint, reqwest::Client::new());
        let resp = provider
            .search(&SearchRequest {
                query: "golang".to_string(),
                max_results: 3,
            })
            .await
            .unwrap();

        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].url, "https://go.dev");
        assert_eq!(resp.results[0].snippet, "Go lang");
        assert_eq!(resp.results[0].page_age.as_deref(), Some("1 day"));
        // Missing age -> None
        assert_eq!(resp.results[1].page_age, None);

        server.abort();
    }

    #[tokio::test]
    async fn missing_web_field_yields_empty() {
        let (endpoint, server) = spawn_mock("{}").await;
        let provider =
            BraveProvider::with_endpoint("k".to_string(), endpoint, reqwest::Client::new());
        let resp = provider
            .search(&SearchRequest {
                query: "x".to_string(),
                max_results: 0,
            })
            .await
            .unwrap();
        assert!(resp.results.is_empty());
        server.abort();
    }
}
