use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, error};

const BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";

#[derive(Debug, Clone)]
pub struct BraveSearchParams {
    pub query: String,
    pub count: Option<u32>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct BraveSearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
    pub page_age: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    description: String,
    page_age: Option<String>,
}

pub async fn search_brave(
    http_client: &Client,
    api_key: &str,
    params: &BraveSearchParams,
) -> Result<Vec<BraveSearchResult>, Box<dyn std::error::Error + Send + Sync>> {
    let count = params.count.unwrap_or(5);

    debug!("Brave Search query: {}", params.query);

    let response = http_client
        .get(BRAVE_SEARCH_URL)
        .query(&[("q", &params.query), ("count", &count.to_string())])
        .header("Accept", "application/json")
        .header("X-Subscription-Token", api_key)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        error!("Brave Search API error: {} {}", status, text);
        return Err(format!("Brave Search failed: {}", status).into());
    }

    let data: BraveSearchResponse = response.json().await?;
    let mut results: Vec<BraveWebResult> = data.web.map(|w| w.results).unwrap_or_default();

    // Filter by allowed domains
    if let Some(ref allowed) = params.allowed_domains {
        if !allowed.is_empty() {
            results.retain(|r| {
                url::Url::parse(&r.url)
                    .ok()
                    .and_then(|u| u.host_str().map(String::from))
                    .is_some_and(|hostname| allowed.iter().any(|d| hostname.contains(d.as_str())))
            });
        }
    }

    // Filter by blocked domains
    if let Some(ref blocked) = params.blocked_domains {
        if !blocked.is_empty() {
            results.retain(|r| {
                url::Url::parse(&r.url)
                    .ok()
                    .and_then(|u| u.host_str().map(String::from))
                    .is_none_or(|hostname| !blocked.iter().any(|d| hostname.contains(d.as_str())))
            });
        }
    }

    debug!("Brave Search returned {} results", results.len());

    Ok(results
        .into_iter()
        .map(|r| BraveSearchResult {
            title: r.title,
            url: r.url,
            description: r.description,
            page_age: r.page_age,
        })
        .collect())
}
