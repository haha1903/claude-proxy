use base64::{engine::general_purpose::STANDARD, Engine};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ROUTING: &str = r#"{"accounts":[{"name":"one","github_token":"fixture-token"}],"clients":[{"api_key":"fixture-client","account":"one"}]}"#;

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claude-proxy"));
    for (name, _) in std::env::vars().filter(|(name, _)| name.starts_with("CLAUDE_PROXY_")) {
        command.env_remove(name);
    }
    command.env("CLAUDE_PROXY__UPSTREAM_URL", "http://127.0.0.1:1");
    command
}

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn json_secret_environment_supports_startup_without_legacy_credentials() {
    let mut command = command();
    command.env("CLAUDE_PROXY__COPILOT_ROUTING", ROUTING);
    assert_authenticated_startup(command);
}

fn assert_authenticated_startup(mut command: Command) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut server = Server(
        command
            .env("CLAUDE_PROXY__BIND_ADDRESS", "127.0.0.1")
            .env("CLAUDE_PROXY__PORT", address.port().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut connection = loop {
        if let Ok(connection) = TcpStream::connect(address) {
            break connection;
        }
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "proxy exited before listening"
        );
        assert!(Instant::now() < deadline, "proxy did not start");
        std::thread::sleep(Duration::from_millis(20));
    };
    connection
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    connection
        .write_all(b"GET /models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    connection.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 401"));
}

fn encrypted_settings() -> (String, String) {
    let plaintext = br#"{"upstream_url":"http://127.0.0.1:1","copilot_pools":{"copilot-1":{"api_key":"fixture-client","github":[{"login":"alice","token":"fixture-token"}]}}}"#;
    let bytes = [7; 32];
    let nonce = [3; 12];
    let key = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &bytes).unwrap());
    let mut ciphertext = plaintext.to_vec();
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce),
        Aad::from(b"claude-proxy-config-v1"),
        &mut ciphertext,
    )
    .unwrap();
    (
        format!(
            "v1.{}",
            STANDARD.encode([nonce.to_vec(), ciphertext].concat())
        ),
        STANDARD.encode(bytes),
    )
}

#[test]
fn encrypted_settings_boot_without_plaintext_credentials() {
    let (envelope, key) = encrypted_settings();
    let mut command = command();
    command
        .env("CLAUDE_PROXY_CONFIG", envelope)
        .env("CLAUDE_PROXY_SECRET", key);
    assert_authenticated_startup(command);
}

#[test]
fn encrypted_settings_errors_do_not_fall_back_or_disclose_values() {
    let (envelope, key) = encrypted_settings();
    for (encrypted, secret) in [
        (Some(envelope.as_str()), None),
        (None, Some(key.as_str())),
        (Some("fixture-secret-invalid"), Some(key.as_str())),
    ] {
        let mut command = command();
        command.env("CLAUDE_PROXY__COPILOT_ROUTING", ROUTING);
        if let Some(value) = encrypted {
            command.env("CLAUDE_PROXY_CONFIG", value);
        }
        if let Some(value) = secret {
            command.env("CLAUDE_PROXY_SECRET", value);
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("configuration"));
        assert!(!error.contains("fixture-secret"));
        assert!(!error.contains(&key));
    }
}

#[test]
fn malformed_environment_secret_fails_closed_without_echoing_credentials() {
    let output = command()
        .env(
            "CLAUDE_PROXY__COPILOT_ROUTING",
            r#"{"accounts":"secret-value"}"#,
        )
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("Invalid copilot_routing JSON"));
    assert!(!error.contains("secret-value"));
}

#[test]
fn plaintext_config_and_rust_encryption_cli_share_the_same_loader() {
    let plaintext = r#"{"upstream_url":"http://127.0.0.1:1","copilot_pools":{"copilot-1":{"api_key":"fixture-client","github":[{"login":"alice","token":"fixture-token"}]}}}"#;
    let mut plain = command();
    plain.env("CLAUDE_PROXY_CONFIG", plaintext);
    assert_authenticated_startup(plain);
    let path = std::env::temp_dir().join(format!("proxy-cli-{}.json", uuid::Uuid::new_v4()));
    let mut child = command()
        .args(["--encrypt-config", path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(plaintext.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    let settings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut encrypted = command();
    for name in ["CLAUDE_PROXY_CONFIG", "CLAUDE_PROXY_SECRET"] {
        encrypted.env(name, settings[name].as_str().unwrap());
    }
    assert_authenticated_startup(encrypted);
    std::fs::write(&path, settings["CLAUDE_PROXY_CONFIG"].as_str().unwrap()).unwrap();
    let mut file = command();
    file.args([
        "--config",
        path.to_str().unwrap(),
        "--secret",
        settings["CLAUDE_PROXY_SECRET"].as_str().unwrap(),
    ]);
    assert_authenticated_startup(file);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn malformed_plaintext_pools_never_disclose_credentials() {
    let output = command().env("CLAUDE_PROXY_CONFIG", r#"{"copilot_pools":{"copilot-1":{"api_key":"fixture-key","github":"fixture-secret"}}}"#).output().unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("Invalid proxy configuration"));
    assert!(!error.contains("fixture-secret"));
}

#[test]
fn copilot_source_requires_an_explicit_upstream_and_accepts_cli_override() {
    let config = r#"{"copilot_pools":{"copilot-1":{"api_key":"fixture-client","github":[{"login":"alice","token":"fixture-token"}]}}}"#;
    let output = command()
        .env_remove("CLAUDE_PROXY__UPSTREAM_URL")
        .env("CLAUDE_PROXY_CONFIG", config)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("upstream_url is required"));
    let mut overridden = command();
    overridden
        .env_remove("CLAUDE_PROXY__UPSTREAM_URL")
        .env("CLAUDE_PROXY_CONFIG", config)
        .args(["--upstream-url", "http://127.0.0.1:1"]);
    assert_authenticated_startup(overridden);
}
