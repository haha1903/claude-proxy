use crate::config::{
    LogLevel, LogRotation, LoggingConfig, ProxyConfig, TlsConfig, UpstreamAuthConfig,
    WebSearchConfig,
};
use crate::{routing, settings};
use clap::Parser;

/// Claude API Proxy - A proxy server for the Claude API with multiple authentication backends
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Encrypt JSON from stdin into a new private settings file, then exit
    #[arg(long, value_name = "OUTPUT", exclusive = true)]
    pub encrypt_config: Option<std::path::PathBuf>,

    /// Path to the configuration file (optional if required params are set via CLI/env)
    #[arg(short, long, value_name = "FILE")]
    config: Option<String>,

    /// Local-only base64 decryption secret; use CLAUDE_PROXY_SECRET for deployment
    #[arg(long, value_name = "SECRET")]
    secret: Option<String>,

    /// Enable verbose (debug) logging
    #[arg(short, long)]
    pub verbose: bool,

    // Server settings
    /// Address to bind the proxy server to
    #[arg(long, value_name = "ADDRESS")]
    bind_address: Option<String>,

    /// Port to listen on
    #[arg(short, long, value_name = "PORT")]
    port: Option<u16>,

    /// Upstream API URL
    #[arg(long, value_name = "URL")]
    upstream_url: Option<String>,

    /// API key that clients must provide to access this proxy
    #[arg(long, value_name = "KEY")]
    client_api_key: Option<String>,

    // Upstream auth settings
    /// Upstream authentication type: api_key, bearer, azure_ad, azure_cli, azure_managed_identity
    #[arg(long, value_name = "TYPE")]
    upstream_auth_type: Option<String>,

    /// API key for upstream authentication (when type=api_key)
    #[arg(long, value_name = "KEY")]
    upstream_api_key: Option<String>,

    /// Bearer token for upstream authentication (when type=bearer)
    #[arg(long, value_name = "TOKEN")]
    upstream_bearer_token: Option<String>,

    /// GitHub OAuth token for upstream authentication (when type=copilot)
    #[arg(long, value_name = "TOKEN")]
    upstream_github_token: Option<String>,

    /// Azure AD tenant ID (when type=azure_ad)
    #[arg(long, value_name = "ID")]
    azure_tenant_id: Option<String>,

    /// Azure AD client ID (when type=azure_ad or azure_managed_identity)
    #[arg(long, value_name = "ID")]
    azure_client_id: Option<String>,

    /// Azure AD client secret (when type=azure_ad)
    #[arg(long, value_name = "SECRET")]
    azure_client_secret: Option<String>,

    /// Azure scope/resource (when type=azure_ad, azure_cli, or azure_managed_identity)
    #[arg(long, value_name = "SCOPE")]
    azure_scope: Option<String>,

    // Logging settings
    /// Path to log file directory
    #[arg(long, value_name = "PATH")]
    log_path: Option<String>,

    /// Log rotation: hourly or daily
    #[arg(long, value_name = "ROTATION")]
    log_rotation: Option<String>,

    /// Log level: trace, debug, info, warn, error
    #[arg(long, value_name = "LEVEL")]
    log_level: Option<String>,

    /// Prefix for log file names
    #[arg(long, value_name = "PREFIX")]
    log_prefix: Option<String>,

    // TLS settings
    /// TLS mode: disabled, manual, or acme
    #[arg(long, value_name = "MODE")]
    tls_mode: Option<String>,

    /// Path to TLS certificate file (manual mode)
    #[arg(long, value_name = "PATH")]
    tls_cert_path: Option<String>,

    /// Path to TLS private key file (manual mode)
    #[arg(long, value_name = "PATH")]
    tls_key_path: Option<String>,

    /// HTTPS listen port (default: 443)
    #[arg(long, value_name = "PORT")]
    https_port: Option<u16>,

    /// ACME contact email
    #[arg(long, value_name = "EMAIL")]
    acme_email: Option<String>,

    /// ACME domains (comma-separated)
    #[arg(long, value_name = "DOMAINS")]
    acme_domains: Option<String>,

    /// ACME directory URL (default: Let's Encrypt production)
    #[arg(long, value_name = "URL")]
    acme_directory_url: Option<String>,

    /// ACME cache directory for certificates and account
    #[arg(long, value_name = "PATH")]
    acme_cache_dir: Option<String>,

    /// HTTP port for ACME HTTP-01 challenges (default: 80)
    #[arg(long, value_name = "PORT")]
    http_challenge_port: Option<u16>,

    /// Custom headers to add to upstream requests (can be specified multiple times)
    #[arg(short = 'H', long = "header", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    headers: Option<Vec<String>>,

    /// Brave Search API key (enables web search emulation)
    #[arg(long, value_name = "KEY")]
    brave_api_key: Option<String>,

    /// Tavily Search API key (enables web search emulation)
    #[arg(long, value_name = "KEY")]
    tavily_api_key: Option<String>,
}

/// Load configuration with layered precedence: CLI > env > file > defaults
pub fn load_config(args: &Args) -> Result<ProxyConfig, Box<dyn std::error::Error>> {
    let file_config = load_source(args)?;
    build_config(args, file_config)
}

fn environment(name: &str) -> Result<Option<String>, &'static str> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err("Configuration environment must contain valid Unicode"),
    }
}

fn load_source(args: &Args) -> Result<Option<settings::LoadedConfig>, Box<dyn std::error::Error>> {
    let secret = args.secret.clone().or(environment("CLAUDE_PROXY_SECRET")?);
    if let Some(path) = &args.config {
        return read_source(path, secret.as_deref()).map(Some);
    }
    if let Some(value) = environment("CLAUDE_PROXY_CONFIG")? {
        return Ok(Some(settings::read(&value, secret.as_deref())?));
    }
    if let Some(path) = environment("CLAUDE_PROXY_CONFIG_FILE")? {
        return read_source(&path, secret.as_deref()).map(Some);
    }
    let paths = [
        Some(std::path::PathBuf::from("config.toml")),
        dirs::config_dir().map(|p| p.join("claude-proxy/config.toml")),
        dirs::home_dir().map(|p| p.join(".claude-proxy.toml")),
    ];
    for path in paths.into_iter().flatten() {
        if path.exists() {
            let source =
                std::fs::read_to_string(path).map_err(|_| "Cannot read configuration file")?;
            return Ok(Some(settings::read(&source, secret.as_deref())?));
        }
    }
    if secret.is_some() {
        return Err("A configuration secret requires a configuration source".into());
    }
    Ok(None)
}

fn read_source(
    path: &str,
    secret: Option<&str>,
) -> Result<settings::LoadedConfig, Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path).map_err(|_| "Cannot read configuration file")?;
    Ok(settings::read(&source, secret)?)
}

/// Trait for parsing environment variable values to target types.
trait FromEnvStr: Sized {
    fn from_env_str(value: &str) -> Option<Self>;
}

impl FromEnvStr for String {
    fn from_env_str(value: &str) -> Option<Self> {
        Some(value.to_string())
    }
}

impl FromEnvStr for u16 {
    fn from_env_str(value: &str) -> Option<Self> {
        value.parse().ok()
    }
}

/// Get value with precedence: CLI > env > file > default
fn get_value<T: Clone + FromEnvStr>(
    cli: Option<T>,
    env_var: &str,
    file: Option<T>,
    default: T,
) -> T {
    if let Some(v) = cli {
        return v;
    }
    if let Ok(v) = std::env::var(env_var) {
        if let Some(parsed) = T::from_env_str(&v) {
            return parsed;
        }
    }
    file.unwrap_or(default)
}

/// Get optional value with precedence: CLI > env > file
fn get_optional_value<T: Clone + FromEnvStr>(
    cli: Option<T>,
    env_var: &str,
    file: Option<T>,
) -> Option<T> {
    if cli.is_some() {
        return cli;
    }
    if let Ok(v) = std::env::var(env_var) {
        if let Some(parsed) = T::from_env_str(&v) {
            return Some(parsed);
        }
    }
    file
}

/// Build the final config by merging CLI args, env vars, file config, and defaults
fn build_config(
    args: &Args,
    source: Option<settings::LoadedConfig>,
) -> Result<ProxyConfig, Box<dyn std::error::Error>> {
    let default_azure_scope = "https://ai.azure.com/.default".to_string();
    let explicit_upstream = source
        .as_ref()
        .is_some_and(|source| source.explicit_upstream);
    let file_config = source.map(|source| source.config);

    // Extract file values
    let (file_bind_address, file_port, file_upstream_url, file_client_api_key, file_upstream_auth) =
        if let Some(ref fc) = file_config {
            (
                Some(fc.bind_address.clone()),
                Some(fc.port),
                Some(fc.upstream_url.clone()).filter(|url| !url.trim().is_empty()),
                Some(fc.client_api_key.clone()),
                fc.upstream_auth.clone(),
            )
        } else {
            (None, None, None, None, None)
        };

    let file_logging = file_config.as_ref().map(|fc| fc.logging.clone());

    // Build basic config values
    let bind_address = get_value(
        args.bind_address.clone(),
        "CLAUDE_PROXY__BIND_ADDRESS",
        file_bind_address,
        "0.0.0.0".to_string(),
    );

    let port = get_value(args.port, "CLAUDE_PROXY__PORT", file_port, 8080);

    let client_api_key = get_optional_value(
        args.client_api_key.clone(),
        "CLAUDE_PROXY__CLIENT_API_KEY",
        file_client_api_key,
    );

    let copilot_routing = match std::env::var("CLAUDE_PROXY__COPILOT_ROUTING") {
        Ok(value) => Some(routing::CopilotRouting::from_json(&value)?),
        Err(std::env::VarError::NotPresent) => file_config
            .as_ref()
            .and_then(|fc| fc.copilot_routing.clone()),
        Err(_) => return Err("CLAUDE_PROXY__COPILOT_ROUTING must be valid Unicode".into()),
    };
    let copilot_pools = file_config.as_ref().and_then(|fc| fc.copilot_pools.clone());
    let upstream_auth = if copilot_routing.is_some() || copilot_pools.is_some() {
        None
    } else {
        Some(build_upstream_auth(
            args,
            file_upstream_auth,
            &default_azure_scope,
        )?)
    };

    let uses_copilot = copilot_routing.is_some()
        || copilot_pools.is_some()
        || matches!(upstream_auth, Some(UpstreamAuthConfig::Copilot { .. }));
    let upstream_url = get_optional_value(
        args.upstream_url.clone(),
        "CLAUDE_PROXY__UPSTREAM_URL",
        file_upstream_url.filter(|_| explicit_upstream || !uses_copilot),
    );

    // Build logging config
    let logging = build_logging_config(args, file_logging);

    // Validate required fields
    let upstream_url = upstream_url.ok_or(
        "upstream_url is required. Set via --upstream-url, CLAUDE_PROXY__UPSTREAM_URL, or config file."
    )?;

    let client_api_key = client_api_key.unwrap_or_default();

    // Build TLS config
    let file_tls = file_config.as_ref().map(|fc| fc.tls.clone());
    let tls = build_tls_config(args, file_tls)?;

    // Parse CLI headers and merge with file config headers
    let file_upstream_headers = file_config
        .as_ref()
        .map(|fc| fc.upstream_headers.clone())
        .unwrap_or_default();
    let upstream_headers = parse_cli_headers(&args.headers, file_upstream_headers);

    // Build web search config: CLI > env > file
    let file_web_search = file_config.as_ref().map(|fc| fc.web_search.clone());
    let brave_api_key = get_optional_value(
        args.brave_api_key.clone(),
        "CLAUDE_PROXY__WEB_SEARCH__BRAVE_API_KEY",
        file_web_search
            .as_ref()
            .and_then(|w| w.brave_api_key.clone()),
    );
    let tavily_api_key = get_optional_value(
        args.tavily_api_key.clone(),
        "CLAUDE_PROXY__WEB_SEARCH__TAVILY_API_KEY",
        file_web_search
            .as_ref()
            .and_then(|w| w.tavily_api_key.clone()),
    );
    let web_search = WebSearchConfig {
        brave_api_key,
        tavily_api_key,
    };

    let config = ProxyConfig {
        bind_address,
        port,
        upstream_url,
        client_api_key,
        upstream_auth,
        copilot_routing,
        copilot_pools,
        upstream_headers,
        logging,
        tls,
        web_search,
    };
    config.validate_auth()?;
    Ok(config)
}

/// Build upstream auth config from CLI args, env vars, and file config
fn build_upstream_auth(
    args: &Args,
    file_auth: Option<UpstreamAuthConfig>,
    default_azure_scope: &str,
) -> Result<UpstreamAuthConfig, Box<dyn std::error::Error>> {
    // Determine auth type: CLI > env > file
    let auth_type = args
        .upstream_auth_type
        .clone()
        .or_else(|| std::env::var("CLAUDE_PROXY__UPSTREAM_AUTH__TYPE").ok())
        .or_else(|| file_auth.as_ref().map(|a| auth_type_name(a).to_string()));

    let auth_type = auth_type.ok_or(
        "upstream_auth.type is required. Set via --upstream-auth-type, CLAUDE_PROXY__UPSTREAM_AUTH__TYPE, or config file."
    )?;

    match auth_type.as_str() {
        "api_key" => {
            let api_key = get_optional_value(
                args.upstream_api_key.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__API_KEY",
                match &file_auth {
                    Some(UpstreamAuthConfig::ApiKey { api_key }) => Some(api_key.clone()),
                    _ => None,
                },
            )
            .ok_or("upstream_auth.api_key is required for api_key auth type. Set via --upstream-api-key, CLAUDE_PROXY__UPSTREAM_AUTH__API_KEY, or config file.")?;

            Ok(UpstreamAuthConfig::ApiKey { api_key })
        }
        "bearer" => {
            let token = get_optional_value(
                args.upstream_bearer_token.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__TOKEN",
                match &file_auth {
                    Some(UpstreamAuthConfig::Bearer { token }) => Some(token.clone()),
                    _ => None,
                },
            )
            .ok_or("upstream_auth.token is required for bearer auth type. Set via --upstream-bearer-token, CLAUDE_PROXY__UPSTREAM_AUTH__TOKEN, or config file.")?;

            Ok(UpstreamAuthConfig::Bearer { token })
        }
        "copilot" => {
            let github_token = get_optional_value(
                args.upstream_github_token.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__GITHUB_TOKEN",
                match &file_auth {
                    Some(UpstreamAuthConfig::Copilot { github_token }) => {
                        Some(github_token.clone())
                    }
                    _ => None,
                },
            )
            .ok_or("upstream_auth.github_token is required for copilot auth type. Set via --upstream-github-token, CLAUDE_PROXY__UPSTREAM_AUTH__GITHUB_TOKEN, or config file.")?;

            Ok(UpstreamAuthConfig::Copilot { github_token })
        }
        "azure_ad" => {
            let tenant_id = get_optional_value(
                args.azure_tenant_id.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__TENANT_ID",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureAd { tenant_id, .. }) => Some(tenant_id.clone()),
                    _ => None,
                },
            )
            .ok_or("upstream_auth.tenant_id is required for azure_ad auth type. Set via --azure-tenant-id, CLAUDE_PROXY__UPSTREAM_AUTH__TENANT_ID, or config file.")?;

            let client_id = get_optional_value(
                args.azure_client_id.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__CLIENT_ID",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureAd { client_id, .. }) => Some(client_id.clone()),
                    _ => None,
                },
            )
            .ok_or("upstream_auth.client_id is required for azure_ad auth type. Set via --azure-client-id, CLAUDE_PROXY__UPSTREAM_AUTH__CLIENT_ID, or config file.")?;

            let client_secret = get_optional_value(
                args.azure_client_secret.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__CLIENT_SECRET",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureAd { client_secret, .. }) => {
                        Some(client_secret.clone())
                    }
                    _ => None,
                },
            )
            .ok_or("upstream_auth.client_secret is required for azure_ad auth type. Set via --azure-client-secret, CLAUDE_PROXY__UPSTREAM_AUTH__CLIENT_SECRET, or config file.")?;

            let scope = get_value(
                args.azure_scope.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__SCOPE",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureAd { scope, .. }) => Some(scope.clone()),
                    _ => None,
                },
                default_azure_scope.to_string(),
            );

            Ok(UpstreamAuthConfig::AzureAd {
                tenant_id,
                client_id,
                client_secret,
                scope,
            })
        }
        "azure_cli" => {
            let scope = get_value(
                args.azure_scope.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__SCOPE",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureCli { scope }) => Some(scope.clone()),
                    _ => None,
                },
                default_azure_scope.to_string(),
            );

            Ok(UpstreamAuthConfig::AzureCli { scope })
        }
        "azure_managed_identity" => {
            let client_id = get_optional_value(
                args.azure_client_id.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__CLIENT_ID",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureManagedIdentity { client_id, .. }) => {
                        client_id.clone()
                    }
                    _ => None,
                },
            );

            let resource = get_value(
                args.azure_scope.clone(),
                "CLAUDE_PROXY__UPSTREAM_AUTH__RESOURCE",
                match &file_auth {
                    Some(UpstreamAuthConfig::AzureManagedIdentity { resource, .. }) => {
                        Some(resource.clone())
                    }
                    _ => None,
                },
                default_azure_scope.to_string(),
            );

            Ok(UpstreamAuthConfig::AzureManagedIdentity { client_id, resource })
        }
        _ => Err(format!(
            "Unknown upstream_auth.type: '{}'. Valid types: api_key, bearer, azure_ad, azure_cli, azure_managed_identity",
            auth_type
        )
        .into()),
    }
}

/// Get the type name for an UpstreamAuthConfig variant
fn auth_type_name(auth: &UpstreamAuthConfig) -> &'static str {
    match auth {
        UpstreamAuthConfig::ApiKey { .. } => "api_key",
        UpstreamAuthConfig::Bearer { .. } => "bearer",
        UpstreamAuthConfig::Copilot { .. } => "copilot",
        UpstreamAuthConfig::AzureAd { .. } => "azure_ad",
        UpstreamAuthConfig::AzureCli { .. } => "azure_cli",
        UpstreamAuthConfig::AzureManagedIdentity { .. } => "azure_managed_identity",
    }
}

/// Build logging config from CLI args, env vars, and file config
fn build_logging_config(args: &Args, file_logging: Option<LoggingConfig>) -> LoggingConfig {
    let file_log_path = file_logging.as_ref().and_then(|l| l.log_path.clone());
    let file_rotation = file_logging.as_ref().map(|l| l.rotation);
    let file_level = file_logging.as_ref().map(|l| l.level);
    let file_prefix = file_logging.as_ref().map(|l| l.log_prefix.clone());

    let log_path = get_optional_value(
        args.log_path.clone(),
        "CLAUDE_PROXY__LOGGING__LOG_PATH",
        file_log_path,
    );

    let rotation = args
        .log_rotation
        .as_ref()
        .and_then(|r| parse_log_rotation(r))
        .or_else(|| {
            std::env::var("CLAUDE_PROXY__LOGGING__ROTATION")
                .ok()
                .and_then(|r| parse_log_rotation(&r))
        })
        .or(file_rotation)
        .unwrap_or_default();

    let level = args
        .log_level
        .as_ref()
        .and_then(|l| parse_log_level(l))
        .or_else(|| {
            std::env::var("CLAUDE_PROXY__LOGGING__LEVEL")
                .ok()
                .and_then(|l| parse_log_level(&l))
        })
        .or(file_level)
        .unwrap_or_default();

    let log_prefix = get_value(
        args.log_prefix.clone(),
        "CLAUDE_PROXY__LOGGING__LOG_PREFIX",
        file_prefix,
        "claude-proxy".to_string(),
    );

    LoggingConfig {
        log_path,
        rotation,
        level,
        log_prefix,
    }
}

fn parse_log_rotation(s: &str) -> Option<LogRotation> {
    match s.to_lowercase().as_str() {
        "hourly" => Some(LogRotation::Hourly),
        "daily" => Some(LogRotation::Daily),
        _ => None,
    }
}

fn parse_log_level(s: &str) -> Option<LogLevel> {
    match s.to_lowercase().as_str() {
        "trace" => Some(LogLevel::Trace),
        "debug" => Some(LogLevel::Debug),
        "info" => Some(LogLevel::Info),
        "warn" => Some(LogLevel::Warn),
        "error" => Some(LogLevel::Error),
        _ => None,
    }
}

/// Parse CLI headers and merge with file config headers.
/// CLI headers take precedence over file config headers for the same key.
fn parse_cli_headers(
    cli_headers: &Option<Vec<String>>,
    file_headers: Vec<(String, String)>,
) -> Vec<(String, String)> {
    let mut result = file_headers;

    if let Some(headers) = cli_headers {
        for header in headers {
            if let Some((key, value)) = header.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();

                // Remove any existing header with the same key (case-insensitive)
                result.retain(|(k, _)| !k.eq_ignore_ascii_case(&key));
                result.push((key, value));
            }
        }
    }

    result
}

/// Build TLS config from CLI args, env vars, and file config
fn build_tls_config(
    args: &Args,
    file_tls: Option<TlsConfig>,
) -> Result<TlsConfig, Box<dyn std::error::Error>> {
    // Determine TLS mode: CLI > env > file > default (disabled)
    let tls_mode = args
        .tls_mode
        .clone()
        .or_else(|| std::env::var("CLAUDE_PROXY__TLS__MODE").ok())
        .or_else(|| file_tls.as_ref().map(|t| tls_mode_name(t).to_string()))
        .unwrap_or_else(|| "disabled".to_string());

    match tls_mode.to_lowercase().as_str() {
        "disabled" => Ok(TlsConfig::Disabled),
        "manual" => {
            let cert_path = get_optional_value(
                args.tls_cert_path.clone(),
                "CLAUDE_PROXY__TLS__CERT_PATH",
                match &file_tls {
                    Some(TlsConfig::Manual { cert_path, .. }) => Some(cert_path.clone()),
                    _ => None,
                },
            )
            .ok_or("tls.cert_path is required for manual TLS mode. Set via --tls-cert-path, CLAUDE_PROXY__TLS__CERT_PATH, or config file.")?;

            let key_path = get_optional_value(
                args.tls_key_path.clone(),
                "CLAUDE_PROXY__TLS__KEY_PATH",
                match &file_tls {
                    Some(TlsConfig::Manual { key_path, .. }) => Some(key_path.clone()),
                    _ => None,
                },
            )
            .ok_or("tls.key_path is required for manual TLS mode. Set via --tls-key-path, CLAUDE_PROXY__TLS__KEY_PATH, or config file.")?;

            let https_port = get_value(
                args.https_port,
                "CLAUDE_PROXY__TLS__HTTPS_PORT",
                match &file_tls {
                    Some(TlsConfig::Manual { https_port, .. }) => Some(*https_port),
                    _ => None,
                },
                443,
            );

            Ok(TlsConfig::Manual {
                cert_path,
                key_path,
                https_port,
            })
        }
        "acme" => {
            let email = get_optional_value(
                args.acme_email.clone(),
                "CLAUDE_PROXY__TLS__EMAIL",
                match &file_tls {
                    Some(TlsConfig::Acme { email, .. }) => Some(email.clone()),
                    _ => None,
                },
            )
            .ok_or("tls.email is required for ACME mode. Set via --acme-email, CLAUDE_PROXY__TLS__EMAIL, or config file.")?;

            // Parse domains from CLI (comma-separated) or use file/env
            let domains = if let Some(domains_str) = args.acme_domains.clone() {
                domains_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect()
            } else if let Ok(domains_str) = std::env::var("CLAUDE_PROXY__TLS__DOMAINS") {
                domains_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect()
            } else {
                match &file_tls {
                    Some(TlsConfig::Acme { domains, .. }) => domains.clone(),
                    _ => Vec::new(),
                }
            };

            if domains.is_empty() {
                return Err("tls.domains is required for ACME mode. Set via --acme-domains, CLAUDE_PROXY__TLS__DOMAINS, or config file.".into());
            }

            let directory_url = get_value(
                args.acme_directory_url.clone(),
                "CLAUDE_PROXY__TLS__DIRECTORY_URL",
                match &file_tls {
                    Some(TlsConfig::Acme { directory_url, .. }) => Some(directory_url.clone()),
                    _ => None,
                },
                "https://acme-v02.api.letsencrypt.org/directory".to_string(),
            );

            let cache_dir = get_value(
                args.acme_cache_dir.clone(),
                "CLAUDE_PROXY__TLS__CACHE_DIR",
                match &file_tls {
                    Some(TlsConfig::Acme { cache_dir, .. }) => Some(cache_dir.clone()),
                    _ => None,
                },
                dirs::data_local_dir()
                    .map(|p| p.join("claude-proxy").join("acme"))
                    .and_then(|p| p.to_str().map(String::from))
                    .unwrap_or_else(|| "/var/lib/claude-proxy/acme".to_string()),
            );

            let https_port = get_value(
                args.https_port,
                "CLAUDE_PROXY__TLS__HTTPS_PORT",
                match &file_tls {
                    Some(TlsConfig::Acme { https_port, .. }) => Some(*https_port),
                    _ => None,
                },
                443,
            );

            let http_challenge_port = get_value(
                args.http_challenge_port,
                "CLAUDE_PROXY__TLS__HTTP_CHALLENGE_PORT",
                match &file_tls {
                    Some(TlsConfig::Acme {
                        http_challenge_port,
                        ..
                    }) => Some(*http_challenge_port),
                    _ => None,
                },
                80,
            );

            Ok(TlsConfig::Acme {
                email,
                domains,
                directory_url,
                cache_dir,
                https_port,
                http_challenge_port,
            })
        }
        _ => Err(format!(
            "Unknown TLS mode: '{}'. Valid modes: disabled, manual, acme",
            tls_mode
        )
        .into()),
    }
}

/// Get the mode name for a TlsConfig variant
fn tls_mode_name(tls: &TlsConfig) -> &'static str {
    match tls {
        TlsConfig::Disabled => "disabled",
        TlsConfig::Manual { .. } => "manual",
        TlsConfig::Acme { .. } => "acme",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_proxy_router;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn copilot_cli_auth_requires_explicit_source_upstream() {
        let source = || {
            settings::read(
            r#"{"client_api_key":"fixture-client","upstream_auth":{"type":"api_key","api_key":"fixture-upstream"}}"#,
            None,
        ).unwrap()
        };
        let default = build_config(&Args::parse_from(["claude-proxy"]), Some(source())).unwrap();
        assert_eq!(default.upstream_url, "https://api.anthropic.com");
        let mut args = Args::parse_from([
            "claude-proxy",
            "--upstream-auth-type",
            "copilot",
            "--upstream-github-token",
            "fixture-token",
        ]);
        assert!(build_config(&args, Some(source())).is_err());
        args.upstream_url = Some("http://127.0.0.1:1".into());
        assert_eq!(
            build_config(&args, Some(source())).unwrap().upstream_url,
            "http://127.0.0.1:1"
        );
    }

    #[tokio::test]
    async fn mapped_config_builds_router_and_requires_client_key() {
        let mut file: ProxyConfig = serde_json::from_value(serde_json::json!({
            "upstream_url":"http://127.0.0.1:1"
        }))
        .unwrap();
        file.copilot_routing = Some(crate::routing::tests::routing());
        let args = Args::parse_from(["claude-proxy"]);
        let config = build_config(
            &args,
            Some(settings::LoadedConfig {
                config: file,
                explicit_upstream: true,
            }),
        )
        .unwrap();
        assert!(config.upstream_auth.is_none());
        let app = build_proxy_router(&config).await.unwrap();
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert!(response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty());
    }

    #[tokio::test]
    async fn legacy_config_still_requires_auth_and_key() {
        let args = Args::parse_from([
            "claude-proxy",
            "--upstream-url",
            "http://127.0.0.1:1",
            "--upstream-auth-type",
            "bearer",
            "--upstream-bearer-token",
            "upstream",
        ]);
        assert!(build_config(&args, None).is_err());
        let mut args = args;
        args.client_api_key = Some("client".into());
        let config = build_config(&args, None).unwrap();
        assert!(config.copilot_routing.is_none());
        assert!(build_proxy_router(&config).await.is_ok());
    }
    #[test]
    fn auth_modes_preserve_file_values_and_cli_overrides() {
        let values = [
            serde_json::json!({"type":"api_key","api_key":"fixture-key"}),
            serde_json::json!({"type":"bearer","token":"fixture-token"}),
            serde_json::json!({"type":"copilot","github_token":"fixture-token"}),
            serde_json::json!({"type":"azure_ad","tenant_id":"tenant","client_id":"client","client_secret":"secret","scope":"custom-scope"}),
            serde_json::json!({"type":"azure_cli","scope":"custom-scope"}),
            serde_json::json!({"type":"azure_managed_identity","client_id":"client","resource":"custom-resource"}),
        ];
        for value in values {
            let auth: UpstreamAuthConfig = serde_json::from_value(value.clone()).unwrap();
            let args = Args::parse_from(["proxy"]);
            let result = build_upstream_auth(&args, Some(auth), "default-scope").unwrap();
            assert_eq!(auth_type_name(&result), value["type"].as_str().unwrap());
            let mut args = Args::parse_from([
                "proxy",
                "--upstream-auth-type",
                value["type"].as_str().unwrap(),
            ]);
            args.upstream_api_key = Some("override-key".into());
            args.upstream_bearer_token = Some("override-token".into());
            args.upstream_github_token = Some("override-token".into());
            args.azure_tenant_id = Some("override-tenant".into());
            args.azure_client_id = Some("override-client".into());
            args.azure_client_secret = Some("override-secret".into());
            args.azure_scope = Some("override-scope".into());
            assert!(build_upstream_auth(&args, None, "default-scope").is_ok());
        }
        for mode in ["api_key", "bearer", "copilot", "azure_ad", "invalid"] {
            let args = Args::parse_from(["proxy", "--upstream-auth-type", mode]);
            assert!(build_upstream_auth(&args, None, "default-scope").is_err());
        }
        let args = Args::parse_from(["proxy"]);
        assert!(build_upstream_auth(&args, None, "default-scope").is_err());
    }

    #[test]
    fn tls_modes_preserve_settings_and_reject_missing_requirements() {
        for value in [
            serde_json::json!({"mode":"disabled"}),
            serde_json::json!({"mode":"manual","cert_path":"cert.pem","key_path":"key.pem","https_port":8443}),
            serde_json::json!({"mode":"acme","email":"user@example.com","domains":["example.com"],"directory_url":"https://example.com/acme","cache_dir":"cache","https_port":8443,"http_challenge_port":8081}),
        ] {
            let tls: TlsConfig = serde_json::from_value(value.clone()).unwrap();
            let args = Args::parse_from(["proxy"]);
            let result = build_tls_config(&args, Some(tls)).unwrap();
            assert_eq!(tls_mode_name(&result), value["mode"].as_str().unwrap());
        }
        let args = Args::parse_from([
            "proxy",
            "--tls-mode",
            "manual",
            "--tls-cert-path",
            "cert.pem",
            "--tls-key-path",
            "key.pem",
            "--https-port",
            "9443",
        ]);
        assert!(matches!(
            build_tls_config(&args, None).unwrap(),
            TlsConfig::Manual {
                https_port: 9443,
                ..
            }
        ));
        let args = Args::parse_from([
            "proxy",
            "--tls-mode",
            "acme",
            "--acme-email",
            "user@example.com",
            "--acme-domains",
            "a.example.com,b.example.com",
        ]);
        assert!(
            matches!(build_tls_config(&args, None).unwrap(), TlsConfig::Acme { domains, .. } if domains.len() == 2)
        );
        for mode in ["manual", "acme", "invalid"] {
            let args = Args::parse_from(["proxy", "--tls-mode", mode]);
            assert!(build_tls_config(&args, None).is_err());
        }
        let args = Args::parse_from([
            "proxy",
            "--tls-mode",
            "acme",
            "--acme-email",
            "user@example.com",
        ]);
        assert!(build_tls_config(&args, None).is_err());
    }

    #[test]
    fn logging_headers_and_value_precedence_are_preserved() {
        let args = Args::parse_from([
            "proxy",
            "--log-path",
            "logs",
            "--log-rotation",
            "hourly",
            "--log-level",
            "debug",
            "--log-prefix",
            "test",
        ]);
        let config = build_logging_config(&args, None);
        assert_eq!(config.rotation, LogRotation::Hourly);
        assert_eq!(config.level, LogLevel::Debug);
        assert_eq!(config.log_prefix, "test");
        assert_eq!(
            build_logging_config(&Args::parse_from(["proxy"]), Some(config))
                .log_path
                .as_deref(),
            Some("logs")
        );
        for name in ["trace", "debug", "info", "warn", "error"] {
            assert!(parse_log_level(name).is_some());
        }
        for name in ["hourly", "daily"] {
            assert!(parse_log_rotation(name).is_some());
        }
        assert!(parse_log_level("invalid").is_none());
        assert!(parse_log_rotation("invalid").is_none());
        let headers = parse_cli_headers(
            &Some(vec!["HEADER=new".into(), "ignored".into()]),
            vec![("header".into(), "old".into())],
        );
        assert_eq!(headers, vec![("HEADER".into(), "new".into())]);
        let name = "PROXY_TEST_PRECEDENCE_UNIQUE";
        std::env::set_var(name, "9");
        assert_eq!(get_value(Some(1u16), name, Some(2), 3), 1);
        assert_eq!(get_value(None::<u16>, name, Some(2), 3), 9);
        assert_eq!(
            get_optional_value(None::<String>, name, None).as_deref(),
            Some("9")
        );
        std::env::set_var(name, "invalid");
        assert_eq!(get_value(None::<u16>, name, Some(2), 3), 2);
        assert_eq!(get_optional_value(None::<u16>, name, Some(4)), Some(4));
        std::env::remove_var(name);
        assert_eq!(get_value(None::<u16>, name, None, 3), 3);
    }
}
