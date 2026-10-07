use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Deserializer};

use crate::config::deserialize_env_string;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopilotRouting {
    pub accounts: Vec<CopilotAccount>,
    pub clients: Vec<CopilotClient>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopilotAccount {
    pub name: String,
    #[serde(deserialize_with = "deserialize_env_string")]
    pub github_token: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopilotClient {
    #[serde(deserialize_with = "deserialize_env_string")]
    pub api_key: String,
    pub account: String,
}

impl fmt::Debug for CopilotRouting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CopilotRouting")
            .field("accounts", &self.accounts.len())
            .field("clients", &self.clients.len())
            .finish()
    }
}

impl CopilotRouting {
    pub fn from_json(value: &str) -> Result<Self, &'static str> {
        serde_json::from_str(value).map_err(|_| "Invalid copilot_routing JSON")
    }

    pub fn validate(&self, headers: &[(String, String)]) -> Result<(), &'static str> {
        if self.accounts.is_empty() || self.clients.is_empty() {
            return Err("copilot_routing requires accounts and clients");
        }
        let mut names = HashSet::new();
        for account in &self.accounts {
            if account.name.trim().is_empty() || !valid_credential(&account.github_token) {
                return Err("copilot_routing account has an empty name or invalid token");
            }
            if !names.insert(&account.name) {
                return Err("copilot_routing account names must be unique");
            }
        }
        let mut keys = HashSet::new();
        for client in &self.clients {
            if !valid_credential(&client.api_key) {
                return Err("copilot_routing client has an invalid API key");
            }
            if !keys.insert(&client.api_key) {
                return Err("copilot_routing client API keys must be unique");
            }
            if !names.contains(&client.account) {
                return Err("copilot_routing client references an unknown account");
            }
        }
        validate_headers(headers)
    }
}

pub fn validate_headers(headers: &[(String, String)]) -> Result<(), &'static str> {
    if headers.iter().any(|(name, _)| {
        ["authorization", "api-key", "x-api-key"]
            .iter()
            .any(|reserved| name.eq_ignore_ascii_case(reserved))
    }) {
        return Err("copilot_routing forbids custom authentication headers");
    }
    Ok(())
}

fn valid_credential(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_graphic())
}

#[derive(Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VaultRecord {
    pub api_key: String,
    #[serde(default = "default_policy")]
    pub policy: String,
    pub github: Vec<GithubAccount>,
}

#[derive(Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GithubAccount {
    pub login: String,
    pub token: String,
}

fn default_policy() -> String {
    "session_hash".into()
}

impl VaultRecord {
    pub fn validate(&self) -> bool {
        let mut identities = HashSet::new();
        valid_credential(&self.api_key)
            && self.policy == "session_hash"
            && !self.github.is_empty()
            && self.github.iter().all(|account| {
                !account.login.is_empty()
                    && account.login.len() <= 100
                    && account
                        .login
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                    && valid_credential(&account.token)
                    && identities.insert(account.login.to_ascii_lowercase())
            })
    }
}

// Length prefixes make the hash input unambiguous. Credentials never define identity.
pub fn rendezvous_score(pool: &str, session: &str, login: &str) -> Vec<u8> {
    let mut hash = ring::digest::Context::new(&ring::digest::SHA256);
    hash.update(b"copilot-session-v1");
    for part in [pool, session, &login.to_ascii_lowercase()] {
        hash.update(&(part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    hash.finish().as_ref().to_vec()
}

// Environment sources provide one JSON string, while TOML provides a table.
// Suppress deserializer details because they can include credential values.
pub fn deserialize_routing<'de, D>(deserializer: D) -> Result<Option<CopilotRouting>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let result = match value {
        serde_json::Value::String(value) => CopilotRouting::from_json(&value),
        value => serde_json::from_value(value).map_err(|_| "Invalid copilot_routing table"),
    };
    result.map(Some).map_err(serde::de::Error::custom)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::ProxyConfig;

    pub(crate) fn routing() -> CopilotRouting {
        CopilotRouting::from_json(r#"{"accounts":[{"name":"one","github_token":"github-one"},{"name":"two","github_token":"github-two"}],"clients":[{"api_key":"client-one","account":"one"},{"api_key":"client-two","account":"two"},{"api_key":"client-shared","account":"one"}]}"#).unwrap()
    }

    #[test]
    fn validates_binding_and_redacts_errors_and_debug() {
        let valid = routing();
        assert!(valid
            .validate(&[("editor-version".into(), "vscode/1.96.0".into())])
            .is_ok());
        let debug = format!("{valid:?}");
        assert!(!debug.contains("github-one"));
        assert!(!debug.contains("client-one"));
        let mutations: Vec<fn(&mut CopilotRouting)> = vec![
            |r| r.accounts.clear(),
            |r| r.clients.clear(),
            |r| r.accounts[0].name.clear(),
            |r| r.accounts[0].github_token = "bad\nheader".into(),
            |r| r.accounts[1].name = "one".into(),
            |r| r.clients[0].api_key = " ".into(),
            |r| r.clients[1].api_key = "client-one".into(),
            |r| r.clients[0].account = "unknown".into(),
        ];
        for mutate in mutations {
            let mut invalid = routing();
            mutate(&mut invalid);
            assert!(invalid.validate(&[]).is_err());
        }
        for header in ["Authorization", "API-Key", "X-API-KEY"] {
            assert!(valid.validate(&[(header.into(), "secret".into())]).is_err());
        }
        for json in [
            "secret",
            r#"{"accounts":"secret","clients":[]}"#,
            r#"{"accounts":[],"clients":[],"round_robin":true}"#,
        ] {
            assert_eq!(
                CopilotRouting::from_json(json).unwrap_err(),
                "Invalid copilot_routing JSON"
            );
        }
    }

    #[test]
    fn reads_toml_tables_and_json_environment_values_without_legacy_auth() {
        let source = r#"
            upstream_url = "https://api.githubcopilot.com"
            [[copilot_routing.accounts]]
            name = "one"
            github_token = "github-one"
            [[copilot_routing.clients]]
            api_key = "client-one"
            account = "one"
        "#;
        let config: ProxyConfig = toml::from_str(source).unwrap();
        assert!(config.upstream_auth.is_none());
        assert!(config.copilot_routing.unwrap().validate(&[]).is_ok());
        let json = r#"{"accounts":[{"name":"one","github_token":"github-one"}],"clients":[{"api_key":"client-one","account":"one"}]}"#;
        let config: ProxyConfig = config::Config::builder()
            .set_override("copilot_routing", json)
            .unwrap()
            .build()
            .unwrap()
            .try_deserialize()
            .unwrap();
        assert_eq!(config.copilot_routing.unwrap().clients[0].account, "one");
        for invalid in [
            serde_json::json!("secret"),
            serde_json::json!({"accounts":"secret"}),
        ] {
            let error = serde_json::from_value::<ProxyConfig>(
                serde_json::json!({"copilot_routing":invalid}),
            )
            .unwrap_err()
            .to_string();
            assert!(!error.contains("secret"));
        }
    }
}
