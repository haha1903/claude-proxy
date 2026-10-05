use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ROUTING: &str = r#"{"accounts":[{"name":"one","github_token":"fixture-token"}],"clients":[{"api_key":"fixture-client","account":"one"}]}"#;

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claude-proxy"));
    for (name, _) in std::env::vars().filter(|(name, _)| name.starts_with("CLAUDE_PROXY__")) {
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
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut server = Server(
        command()
            .env("CLAUDE_PROXY__COPILOT_ROUTING", ROUTING)
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
