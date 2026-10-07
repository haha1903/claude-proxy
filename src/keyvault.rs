use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use futures::{stream, StreamExt};
use reqwest::{Client, Url};
use serde_json::Value;
use tracing::{info, warn};

use crate::{
    middleware::ApiKeyValidatorState,
    routing::{GithubAccount, VaultRecord as Record},
};

const REFRESH: Duration = Duration::from_secs(60);
const LIMIT: usize = 1024 * 1024;

pub fn validate_vault_url(value: &str) -> Result<(), &'static str> {
    let url = Url::parse(value).map_err(|_| "Invalid Key Vault URL")?;
    let host = url.host_str().unwrap_or_default();
    let name = host.strip_suffix(".vault.azure.net").unwrap_or_default();
    if url.scheme() != "https"
        || name.is_empty()
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("Invalid Key Vault URL");
    }
    Ok(())
}

fn number(name: &str) -> Option<u32> {
    let suffix = name.strip_prefix("copilot-")?;
    if suffix.starts_with('0') || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

fn credential(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_graphic())
}

fn enabled(attributes: &Value, now: i64) -> bool {
    attributes.get("enabled").and_then(Value::as_bool) != Some(false)
        && attributes
            .get("nbf")
            .and_then(Value::as_i64)
            .is_none_or(|t| t <= now)
        && attributes
            .get("exp")
            .and_then(Value::as_i64)
            .is_none_or(|t| t > now)
}

async fn json(request: reqwest::RequestBuilder) -> Result<Value, &'static str> {
    let mut response = request.send().await.map_err(|_| "Remote request failed")?;
    if !response.status().is_success() {
        return Err("Remote request returned a non-success status");
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "Remote response failed")?
    {
        if bytes.len() + chunk.len() > LIMIT {
            return Err("Remote response too large");
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid remote JSON")
}

struct Vault {
    url: Url,
    http: Client,
    identity_endpoint: String,
    identity_header: String,
    github: String,
    records: HashMap<String, Record>,
}

impl Vault {
    fn new(url: &str) -> Result<Self, &'static str> {
        validate_vault_url(url)?;
        Ok(Self {
            url: Url::parse(url).map_err(|_| "Invalid Key Vault URL")?,
            http: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .no_proxy()
                .timeout(Duration::from_secs(25))
                .build()
                .map_err(|_| "Cannot create Vault client")?,
            identity_endpoint: std::env::var("IDENTITY_ENDPOINT")
                .map_err(|_| "Container Apps managed identity is required")?,
            identity_header: std::env::var("IDENTITY_HEADER")
                .map_err(|_| "Container Apps identity header is required")?,
            github: "https://api.github.com".into(),
            records: HashMap::new(),
        })
    }

    async fn token(&self) -> Result<String, &'static str> {
        let result = json(
            self.http
                .get(&self.identity_endpoint)
                .header("X-IDENTITY-HEADER", &self.identity_header)
                .query(&[
                    ("api-version", "2019-08-01"),
                    ("resource", "https://vault.azure.net"),
                ]),
        )
        .await?;
        result["access_token"]
            .as_str()
            .filter(|s| credential(s))
            .map(str::to_owned)
            .ok_or("Invalid managed identity response")
    }

    async fn list(&self, token: &str) -> Result<Vec<String>, &'static str> {
        let mut next = Some(
            self.url
                .join("secrets?api-version=7.4")
                .map_err(|_| "Invalid Vault URL")?,
        );
        let mut names = HashSet::new();
        let mut visited = HashSet::new();
        while let Some(url) = next.take() {
            // Never send a Vault token to a continuation URL on another origin.
            if url.origin() != self.url.origin()
                || url.path() != "/secrets"
                || !url.username().is_empty()
                || url.password().is_some()
                || url.fragment().is_some()
                || visited.len() >= 100
                || !visited.insert(url.to_string())
            {
                return Err("Invalid Vault continuation");
            }
            let page = json(self.http.get(url).bearer_auth(token)).await?;
            for item in page["value"].as_array().ok_or("Invalid Vault list")? {
                let id = Url::parse(item["id"].as_str().ok_or("Invalid secret ID")?)
                    .map_err(|_| "Invalid secret ID")?;
                if id.origin() != self.url.origin() {
                    return Err("Invalid secret origin");
                }
                let Some(name) = id.path().strip_prefix("/secrets/") else {
                    return Err("Invalid secret path");
                };
                if number(name).is_some()
                    && enabled(&item["attributes"], chrono::Utc::now().timestamp())
                {
                    names.insert(name.to_owned());
                }
            }
            next = match page.get("nextLink") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) => {
                    Some(Url::parse(value).map_err(|_| "Invalid Vault continuation")?)
                }
                _ => return Err("Invalid Vault continuation"),
            };
        }
        Ok(names.into_iter().collect())
    }

    async fn identity_matches(&self, record: &GithubAccount) -> Result<bool, &'static str> {
        let get = |path: &str| {
            self.http
                .get(format!("{}{path}", self.github))
                .header("Authorization", format!("token {}", record.token))
                .header("User-Agent", "GithubCopilot/1.96.0")
                .header("editor-version", "vscode/1.96.0")
        };
        let mut result = json(get("/copilot_internal/user")).await?;
        if result.get("login").is_none() {
            result = json(get("/user")).await?;
        }
        Ok(result["login"]
            .as_str()
            .is_some_and(|login| login.eq_ignore_ascii_case(&record.login)))
    }

    async fn record(&self, name: &str, token: &str) -> Result<Option<Record>, &'static str> {
        let url = self
            .url
            .join(&format!("secrets/{name}?api-version=7.4"))
            .map_err(|_| "Invalid secret URL")?;
        let value = json(self.http.get(url).bearer_auth(token)).await?;
        if !enabled(&value["attributes"], chrono::Utc::now().timestamp()) {
            return Ok(None);
        }
        let record: Record = match value["value"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
        {
            Some(record) => record,
            None => {
                warn!(
                    secret = name,
                    "Invalid Copilot record; disabled until corrected"
                );
                return Ok(None);
            }
        };
        if !record.validate() {
            warn!(secret = name, "Invalid Copilot fields; record disabled");
            return Ok(None);
        }
        for account in &record.github {
            let verified = self.records.values().flat_map(|r| &r.github).any(|old| {
                old.login.eq_ignore_ascii_case(&account.login) && old.token == account.token
            });
            if !verified && !self.identity_matches(account).await? {
                warn!(secret = name, "GitHub identity mismatch; record disabled");
                return Ok(None);
            }
        }
        Ok(Some(record))
    }

    async fn refresh(&mut self, state: &ApiKeyValidatorState) -> Result<(), &'static str> {
        let token = self.token().await?;
        let names = self.list(&token).await?;
        let vault = &*self;
        let credential = &token;
        let fetched = stream::iter(names.into_iter().map(|name| async move {
            let result = vault.record(&name, credential).await;
            (name, result)
        }))
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await;
        let mut records = HashMap::new();
        for (name, result) in fetched {
            match result {
                Ok(Some(record)) => {
                    records.insert(name, record);
                }
                Ok(None) => {}
                Err(reason) => {
                    warn!(
                        secret = name,
                        reason,
                        "Copilot record unavailable; retaining previous verified value if present"
                    );
                    if let Some(previous) = self.records.get(&name) {
                        records.insert(name, previous.clone());
                    }
                }
            }
        }
        let mut keys = HashMap::new();
        for record in records.values() {
            *keys.entry(record.api_key.clone()).or_insert(0) += 1;
        }
        records.retain(|name, record| {
            let unique = keys[&record.api_key] == 1;
            if !unique {
                warn!(secret = name, "Duplicate Copilot binding; record disabled");
            }
            unique
        });
        state.replace_records(&records)?;
        if self.records != records {
            info!(records = records.len(), "Copilot Vault routing updated");
        }
        self.records = records;
        Ok(())
    }
}

pub async fn start(url: &str, state: ApiKeyValidatorState) -> Result<(), &'static str> {
    let mut vault = Vault::new(url)?;
    tokio::time::timeout(Duration::from_secs(120), vault.refresh(&state))
        .await
        .map_err(|_| "Initial Vault load timed out")??;
    if vault.records.is_empty() {
        return Err("No verified Copilot records in Key Vault");
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval_at(tokio::time::Instant::now() + REFRESH, REFRESH);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            match tokio::time::timeout(Duration::from_secs(120), vault.refresh(&state)).await {
                Ok(Ok(())) => {}
                _ => warn!("Copilot Vault refresh failed; retaining previous routing"),
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests;
