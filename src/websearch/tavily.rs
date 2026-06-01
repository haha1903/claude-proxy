//! Tavily Search API provider.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::{
    parse_search_response, SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult,
};

const TAVILY_SEARCH_ENDPOINT: &str = "https://api.tavily.com/search";
const TAVILY_SEARCH_DEPTH_BASIC: &str = "basic";

/// Web search via the Tavily Search API.
pub struct TavilyProvider {
    api_key: String,
    endpoint: String,
    http_client: reqwest::Client,
}

impl TavilyProvider {
    /// Create a Tavily provider using the production endpoint.
    pub fn new(api_key: String, http_client: reqwest::Client) -> Self {
        Self {
            api_key,
            endpoint: TAVILY_SEARCH_ENDPOINT.to_string(),
            http_client,
        }
    }

    /// Create a Tavily provider with a custom endpoint (used by tests).
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
impl SearchProvider for TavilyProvider {
    fn name(&self) -> &'static str {
        "tavily"
    }

    async fn search(&self, req: &SearchRequest) -> Result<SearchResponse, SearchError> {
        let max_results = req.effective_max_results();

        debug!("Tavily search query: {}", req.query);

        let payload = TavilyRequest {
            api_key: &self.api_key,
            query: &req.query,
            max_results,
            search_depth: TAVILY_SEARCH_DEPTH_BASIC,
        };

        let response = self
            .http_client
            .post(&self.endpoint)
            .json(&payload)
            .send()
            .await?;

        let raw: TavilyResponse = parse_search_response(response).await?;

        let results = raw
            .results
            .into_iter()
            .map(|r| SearchResult {
                url: r.url,
                title: r.title,
                snippet: r.content,
                page_age: None,
            })
            .collect();

        Ok(SearchResponse { results })
    }
}

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    max_results: u32,
    search_depth: &'a str,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    async fn spawn_mock(body: &'static str) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/search",
            post(move || async move { ([("content-type", "application/json")], body) }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{}/search", addr), handle)
    }

    #[tokio::test]
    async fn parses_results_without_page_age() {
        let body = r#"{"results":[
            {"url":"https://go.dev","title":"Go","content":"Go programming language","score":0.95}
        ]}"#;
        let (endpoint, server) = spawn_mock(body).await;

        let provider =
            TavilyProvider::with_endpoint("k".to_string(), endpoint, reqwest::Client::new());
        let resp = provider
            .search(&SearchRequest {
                query: "golang".to_string(),
                max_results: 3,
            })
            .await
            .unwrap();

        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].url, "https://go.dev");
        assert_eq!(resp.results[0].snippet, "Go programming language");
        assert_eq!(resp.results[0].page_age, None);

        server.abort();
    }

    #[tokio::test]
    async fn empty_results() {
        let (endpoint, server) = spawn_mock(r#"{"results":[]}"#).await;
        let provider =
            TavilyProvider::with_endpoint("k".to_string(), endpoint, reqwest::Client::new());
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
