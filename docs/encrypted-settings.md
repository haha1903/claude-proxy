# Encrypted Container Apps settings

The proxy can load its JSON configuration from `CLAUDE_PROXY_CONFIG`
and `CLAUDE_PROXY_SECRET`. Configure both as Container Apps secrets and
reference them from environment variables. No Key Vault access is required.

The envelope uses AES-256-GCM with a random 32-byte key, a random 12-byte nonce,
and the authenticated context `claude-proxy-config-v1`. Its format is
`v1.<base64(nonce || ciphertext || 16-byte tag)>`. The key is standard base64.
Invalid, incomplete, or tampered settings prevent startup without logging values.
This hides plaintext in settings. Anyone able to read both secrets can decrypt it.

When `CLAUDE_PROXY_SECRET` (or `--secret`) is present, config must be encrypted.
Without a secret, config is plaintext JSON or TOML. There is no automatic fallback.
`CLAUDE_PROXY_CONFIG` contains config directly. `--config FILE` takes precedence
and reads the same plaintext or encrypted content from a file. Existing command-line
and `CLAUDE_PROXY__*` environment overrides retain their precedence. Configure
only one of `copilot_pools` and `copilot_routing`. Set an explicit upstream URL
for Copilot. Configuration is loaded once at startup, with no remote refresh.

Example plaintext structure, using placeholders only:

```json
{
  "upstream_url": "https://api.githubcopilot.com",
  "copilot_pools": {
    "copilot-1": {
      "api_key": "<existing client key>",
      "github": [{"login": "<existing login>", "token": "<existing token>"}]
    },
    "copilot-5": {
      "api_key": "<pool client key>",
      "policy": "session_hash",
      "github": [
        {"login": "<first login>", "token": "<first token>"},
        {"login": "<second login>", "token": "<second token>"}
      ]
    }
  },
  "web_search": {"tavily_api_key": "<existing search key>"}
}
```

Use `claude-proxy --encrypt-config <private-settings.json>` with JSON on stdin.
The same Rust binary encrypts and decrypts. No Python dependency is needed.
The output is created with mode 0600 on Unix and is never overwritten. On Windows,
the file inherits the containing directory's ACL, so use a private directory.
Keep it outside Git and the Docker build context.
Upload the two values as separate app secrets. Deploy a new revision or restart
the active revision after changing secrets. Pools load once per process.
Preserve pool names to preserve the existing session hash. Single-member pools
stay fixed. Multi-member pools require a Codex or Claude Code session header.
