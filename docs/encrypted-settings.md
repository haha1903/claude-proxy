# Encrypted Container Apps settings

The proxy can load its JSON configuration from `CLAUDE_PROXY_ENCRYPTED_CONFIG`
and `CLAUDE_PROXY_CONFIG_KEY`. Configure both as Container Apps secrets and
reference them from environment variables. No Key Vault access is required.

The envelope uses AES-256-GCM with a random 32-byte key, a random 12-byte nonce,
and the authenticated context `claude-proxy-config-v1`. Its format is
`v1.<base64(nonce || ciphertext || 16-byte tag)>`. The key is standard base64.
Invalid, incomplete, or tampered settings prevent startup without logging values.
This hides plaintext in settings. Anyone able to read both secrets can decrypt it.

The encrypted document replaces the configuration file. Existing command-line
and `CLAUDE_PROXY__*` environment overrides retain their precedence. Configure
only one of `copilot_pools`, `copilot_routing`, and `copilot_vault_url`.

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

Use `tools/encrypt_config.py --output <private-settings.json>` with JSON on stdin
and Python's `cryptography` package installed. The output is created with mode
0600 and is never overwritten. Keep it outside Git and the Docker build context.
Upload the two values as separate app secrets. Deploy a new revision or restart
the active revision after changing secrets. Pools load once per process.
Preserve pool names to preserve the existing session hash. Single-member pools
stay fixed. Multi-member pools require a Codex or Claude Code session header.
