mod acme;
mod auth;
mod cli;
mod config;
mod copilot_responses;
mod middleware;
mod proxy;
mod routing;
mod settings;
mod tls;
mod web_search_emulation;
mod websearch;

use axum::{middleware as axum_middleware, routing::any, Router};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, Layer};

use crate::config::{LogLevel, LogRotation, ProxyConfig, TlsConfig, WebSearchConfig};
use crate::middleware::{validate_client_api_key, ApiKeyValidatorState};
use crate::proxy::{build_http_client, proxy_handler, ProxyState};
use crate::websearch::{BraveProvider, SearchProvider, TavilyProvider, WebSearchManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments first (before logging init)
    let args = cli::Args::parse();
    if let Some(path) = &args.encrypt_config {
        settings::write_settings(std::io::stdin().lock(), path)?;
        return Ok(());
    }

    // Load configuration first (needed for logging setup)
    let config = cli::load_config(&args)?;

    // Initialize logging with appropriate level
    // --verbose flag overrides config setting to DEBUG
    let log_level = if args.verbose {
        Level::DEBUG
    } else {
        match config.logging.level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    };

    // Set up logging based on configuration
    let _guard = init_logging(log_level, &config)?;

    info!("Starting Claude API Proxy");
    info!("Upstream URL: {}", config.upstream_url);
    if config.web_search.brave_api_key.is_some() || config.web_search.tavily_api_key.is_some() {
        info!("Web search emulation enabled");
    }

    // Build the main proxy router
    let app = build_proxy_router(&config).await?;

    // Start server(s) based on TLS configuration
    match &config.tls {
        TlsConfig::Disabled => {
            info!(
                "TLS disabled, starting HTTP server on {}:{}",
                config.bind_address, config.port
            );
            start_http_server(app, &config).await?;
        }
        TlsConfig::Manual {
            cert_path,
            key_path,
            https_port,
        } => {
            info!(
                "TLS enabled (manual mode), starting HTTPS server on {}:{}",
                config.bind_address, https_port
            );
            start_https_server_manual(app, &config, cert_path, key_path, *https_port).await?;
        }
        TlsConfig::Acme {
            email,
            domains,
            directory_url,
            cache_dir,
            https_port,
            http_challenge_port,
        } => {
            info!("TLS enabled (ACME mode) for domains: {:?}", domains);
            info!(
                "HTTPS server on {}:{}, HTTP challenge server on {}:{}",
                config.bind_address, https_port, config.bind_address, http_challenge_port
            );
            start_https_server_acme(
                app,
                &config,
                email,
                domains,
                directory_url,
                cache_dir,
                *https_port,
                *http_challenge_port,
            )
            .await?;
        }
    }

    Ok(())
}

/// Build the main proxy router
async fn build_proxy_router(config: &ProxyConfig) -> Result<Router, Box<dyn std::error::Error>> {
    let api_key_state = ApiKeyValidatorState::from_config(config)?;

    // Create HTTP client for upstream requests
    let http_client = build_http_client().expect("Failed to create HTTP client");

    // Build the web search manager (None when no provider keys are configured)
    let web_search = build_web_search_manager(&config.web_search);
    if let Some(ref mgr) = web_search {
        info!("Web search emulation providers: {}", mgr.provider_names());
    }

    // Create proxy state
    let proxy_state = ProxyState {
        upstream_url: config.upstream_url.clone(),
        http_client: Arc::new(RwLock::new(http_client)),
        upstream_headers: config.upstream_headers.clone(),
        web_search,
        copilot_responses: config.uses_copilot(),
    };

    // Build the router
    Ok(Router::new()
        .route("/{*path}", any(proxy_handler))
        .route("/", any(proxy_handler))
        .layer(axum_middleware::from_fn_with_state(
            api_key_state,
            validate_client_api_key,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(proxy_state))
}

/// Build a web search manager from configured provider keys, or `None` if no
/// keys are set. Providers are added in a fixed order (Brave, then Tavily) so
/// round-robin starts deterministically.
fn build_web_search_manager(config: &WebSearchConfig) -> Option<Arc<WebSearchManager>> {
    let search_client = reqwest::Client::new();
    let mut providers: Vec<Arc<dyn SearchProvider>> = Vec::new();

    if let Some(ref key) = config.brave_api_key {
        providers.push(Arc::new(BraveProvider::new(
            key.clone(),
            search_client.clone(),
        )));
    }
    if let Some(ref key) = config.tavily_api_key {
        providers.push(Arc::new(TavilyProvider::new(
            key.clone(),
            search_client.clone(),
        )));
    }

    WebSearchManager::new(providers).map(Arc::new)
}

/// Start plain HTTP server (existing behavior when TLS is disabled)
async fn start_http_server(
    app: Router,
    config: &ProxyConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = format!("{}:{}", config.bind_address, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!(
        "Claude API Proxy is ready to accept connections on http://{}",
        addr
    );
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start HTTPS server with manual certificate files
async fn start_https_server_manual(
    app: Router,
    config: &ProxyConfig,
    cert_path: &str,
    key_path: &str,
    https_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let rustls_config = tls::setup_manual_tls(cert_path, key_path).await?;
    let addr: SocketAddr = format!("{}:{}", config.bind_address, https_port).parse()?;

    info!(
        "Claude API Proxy is ready to accept connections on https://{}",
        addr
    );

    axum_server::bind_rustls(addr, rustls_config)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

/// Start HTTPS server with ACME certificate provisioning
/// Also starts a separate HTTP server for ACME HTTP-01 challenges
#[allow(clippy::too_many_arguments)]
async fn start_https_server_acme(
    app: Router,
    config: &ProxyConfig,
    email: &str,
    domains: &[String],
    directory_url: &str,
    cache_dir: &str,
    https_port: u16,
    http_challenge_port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    // Create ACME manager
    let manager =
        Arc::new(acme::AcmeManager::new(email, domains.to_vec(), directory_url, cache_dir).await?);

    let rustls_config = manager.rustls_config();
    let challenge_state = manager.challenge_state();

    // Start certificate renewal background task
    let _renewal_handle = manager.clone().start_renewal_loop();

    // Build HTTP-01 challenge server
    let challenge_app = acme::build_challenge_router(challenge_state);
    let challenge_addr: SocketAddr =
        format!("{}:{}", config.bind_address, http_challenge_port).parse()?;

    // Build HTTPS server address
    let https_addr: SocketAddr = format!("{}:{}", config.bind_address, https_port).parse()?;

    info!(
        "Starting ACME HTTP-01 challenge server on http://{}",
        challenge_addr
    );
    info!(
        "Claude API Proxy is ready to accept connections on https://{}",
        https_addr
    );

    // Run both servers concurrently
    tokio::select! {
        result = start_challenge_server(challenge_app, challenge_addr) => {
            result?;
        }
        result = start_tls_server(app, https_addr, rustls_config) => {
            result?;
        }
    }

    Ok(())
}

/// Start the HTTP-01 challenge server
async fn start_challenge_server(
    app: Router,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the TLS server
async fn start_tls_server(
    app: Router,
    addr: SocketAddr,
    config: axum_server::tls_rustls::RustlsConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

/// Initialize logging with optional file rotation.
/// Returns a guard that must be kept alive for the duration of the program
/// to ensure logs are flushed properly.
fn init_logging(
    log_level: Level,
    config: &ProxyConfig,
) -> Result<Option<tracing_appender::non_blocking::WorkerGuard>, Box<dyn std::error::Error>> {
    match &config.logging.log_path {
        Some(log_path) => {
            // Create the log directory if it doesn't exist
            std::fs::create_dir_all(log_path)?;

            // Determine rotation frequency
            let rotation = match config.logging.rotation {
                LogRotation::Hourly => Rotation::HOURLY,
                LogRotation::Daily => Rotation::DAILY,
            };

            // Create rolling file appender
            let file_appender =
                RollingFileAppender::new(rotation, log_path, &config.logging.log_prefix);

            // Create non-blocking writer
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

            // Set up subscriber with both stdout and file output
            tracing_subscriber::registry()
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_writer(std::io::stdout)
                        .with_filter(tracing_subscriber::filter::LevelFilter::from_level(
                            log_level,
                        )),
                )
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_ansi(false)
                        .with_writer(non_blocking)
                        .with_filter(tracing_subscriber::filter::LevelFilter::from_level(
                            log_level,
                        )),
                )
                .init();

            Ok(Some(guard))
        }
        None => {
            // No log path specified, only log to stdout
            tracing_subscriber::registry()
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_writer(std::io::stdout)
                        .with_filter(tracing_subscriber::filter::LevelFilter::from_level(
                            log_level,
                        )),
                )
                .init();

            Ok(None)
        }
    }
}
