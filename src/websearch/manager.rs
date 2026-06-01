//! Multi-provider search manager: round-robin selection with failover.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tracing::warn;

use super::{SearchProvider, SearchRequest, SearchResponse};

/// Manages one or more search providers, balancing load round-robin and
/// failing over to the next provider when one errors.
pub struct WebSearchManager {
    providers: Vec<Arc<dyn SearchProvider>>,
    cursor: AtomicUsize,
}

impl WebSearchManager {
    /// Create a manager. Returns `None` if no providers are configured, so the
    /// caller can treat "no providers" as "web search disabled".
    pub fn new(providers: Vec<Arc<dyn SearchProvider>>) -> Option<Self> {
        if providers.is_empty() {
            return None;
        }
        Some(Self {
            providers,
            cursor: AtomicUsize::new(0),
        })
    }

    /// Comma-separated provider names, for logging.
    pub fn provider_names(&self) -> String {
        self.providers
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Execute a search. Starts at the next round-robin provider and tries each
    /// in rotation until one succeeds. Returns the response and the name of the
    /// provider that served it, or `None` if every provider failed.
    pub async fn search(&self, req: &SearchRequest) -> Option<(SearchResponse, &'static str)> {
        let n = self.providers.len();
        // fetch_add wraps on overflow in release; modulo keeps it in range.
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % n;

        for offset in 0..n {
            let provider = &self.providers[(start + offset) % n];
            match provider.search(req).await {
                Ok(resp) => return Some((resp, provider.name())),
                Err(e) => {
                    warn!(
                        "web search provider '{}' failed: {}; trying next",
                        provider.name(),
                        e
                    );
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::websearch::{SearchError, SearchResult};
    use async_trait::async_trait;
    use std::sync::atomic::AtomicU32;

    /// A provider that records how many times it was called and returns a
    /// configurable success/failure.
    struct StubProvider {
        name: &'static str,
        calls: Arc<AtomicU32>,
        succeed: bool,
    }

    #[async_trait]
    impl SearchProvider for StubProvider {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn search(&self, _req: &SearchRequest) -> Result<SearchResponse, SearchError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.succeed {
                Ok(SearchResponse {
                    results: vec![SearchResult {
                        url: format!("https://{}", self.name),
                        title: self.name.to_string(),
                        snippet: String::new(),
                        page_age: None,
                    }],
                })
            } else {
                Err(SearchError::Decode("stub failure".to_string()))
            }
        }
    }

    fn req() -> SearchRequest {
        SearchRequest {
            query: "q".to_string(),
            max_results: 5,
        }
    }

    #[test]
    fn new_returns_none_when_empty() {
        assert!(WebSearchManager::new(vec![]).is_none());
    }

    #[tokio::test]
    async fn round_robin_alternates_between_providers() {
        let a_calls = Arc::new(AtomicU32::new(0));
        let b_calls = Arc::new(AtomicU32::new(0));
        let mgr = WebSearchManager::new(vec![
            Arc::new(StubProvider {
                name: "a",
                calls: a_calls.clone(),
                succeed: true,
            }),
            Arc::new(StubProvider {
                name: "b",
                calls: b_calls.clone(),
                succeed: true,
            }),
        ])
        .unwrap();

        // First call -> provider a, second -> b, third -> a.
        assert_eq!(mgr.search(&req()).await.unwrap().1, "a");
        assert_eq!(mgr.search(&req()).await.unwrap().1, "b");
        assert_eq!(mgr.search(&req()).await.unwrap().1, "a");
        assert_eq!(a_calls.load(Ordering::SeqCst), 2);
        assert_eq!(b_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fails_over_to_next_provider() {
        let mgr = WebSearchManager::new(vec![
            Arc::new(StubProvider {
                name: "bad",
                calls: Arc::new(AtomicU32::new(0)),
                succeed: false,
            }),
            Arc::new(StubProvider {
                name: "good",
                calls: Arc::new(AtomicU32::new(0)),
                succeed: true,
            }),
        ])
        .unwrap();

        // First round-robin pick is "bad", which fails -> failover to "good".
        let (_, name) = mgr.search(&req()).await.unwrap();
        assert_eq!(name, "good");
    }

    #[tokio::test]
    async fn returns_none_when_all_fail() {
        let mgr = WebSearchManager::new(vec![Arc::new(StubProvider {
            name: "bad",
            calls: Arc::new(AtomicU32::new(0)),
            succeed: false,
        })])
        .unwrap();
        assert!(mgr.search(&req()).await.is_none());
    }
}
