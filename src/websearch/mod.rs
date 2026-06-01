//! Web search provider abstraction.
//!
//! Defines a `SearchProvider` trait and shared request/response types. Concrete
//! providers (Brave, Tavily) live in submodules. The `WebSearchManager`
//! load-balances across configured providers with round-robin + failover.

mod brave;
mod manager;
mod tavily;

pub use brave::BraveProvider;
pub use manager::WebSearchManager;
pub use tavily::TavilyProvider;

use async_trait::async_trait;
use thiserror::Error;

/// Default number of results to request when the caller does not specify.
pub const DEFAULT_MAX_RESULTS: u32 = 5;

/// A single web search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    /// Short snippet / description of the page content.
    pub snippet: String,
    /// Page freshness, if the provider reports it (Brave does, Tavily does not).
    pub page_age: Option<String>,
}

/// Describes a web search to perform.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    /// Maximum results to return; falls back to `DEFAULT_MAX_RESULTS` if zero.
    pub max_results: u32,
}

impl SearchRequest {
    /// Requested result count, substituting the default when unset (zero).
    fn effective_max_results(&self) -> u32 {
        if self.max_results == 0 {
            DEFAULT_MAX_RESULTS
        } else {
            self.max_results
        }
    }
}

/// The results of a web search.
#[derive(Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

/// Errors a provider can return.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("provider returned status {status}: {body}")]
    Status { status: u16, body: String },

    #[error("failed to decode response: {0}")]
    Decode(String),
}

/// A web search backend (Brave, Tavily, ...).
#[async_trait]
pub trait SearchProvider: Send + Sync {
    /// Stable provider identifier ("brave" / "tavily").
    fn name(&self) -> &'static str;

    /// Execute a web search.
    async fn search(&self, req: &SearchRequest) -> Result<SearchResponse, SearchError>;
}

/// Check a provider response's status and deserialize its JSON body, mapping
/// failures to `SearchError`. Shared by all providers.
async fn parse_search_response<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, SearchError> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(SearchError::Status {
            status: status.as_u16(),
            body: truncate_body(&body),
        });
    }
    response
        .json()
        .await
        .map_err(|e| SearchError::Decode(e.to_string()))
}

/// Truncate a response body for inclusion in error messages.
fn truncate_body(body: &str) -> String {
    const MAX: usize = 200;
    if body.len() <= MAX {
        body.to_string()
    } else {
        format!("{}...(truncated)", &body[..MAX])
    }
}
