# Copilot records in Azure Key Vault

Set `CLAUDE_PROXY__COPILOT_VAULT_URL=https://<vault>.vault.azure.net/` on the
Container App. Enable its managed identity and grant Key Vault Secrets User
at Vault scope. This role can read all secrets in that Vault. The application
only loads names matching `copilot-[1-9][0-9]*` (up to a 32-bit number).
Do not set `CLAUDE_PROXY__COPILOT_ROUTING` together with the Vault URL.

Create a secret such as `copilot-1` with this JSON value:

```json
{
  "api_key": "<independent-proxy-access-key>",
  "github": [
    {"login": "<github-login>", "token": "<github-token>"}
  ]
}
```

The optional `policy` defaults to `session_hash`, currently the only supported
policy. Multiple `github` entries form a pool. Each entry must identify a
different account, and API keys must be unique across secrets. Accounts may
appear in several pools. Secret names do not contain account identities.

The proxy polls every 60 seconds and validates changed GitHub credentials before
publishing an atomic configuration snapshot. In-flight requests retain their
selected account. Network failures retain the last verified configuration in
memory. Invalid, disabled, expired, and deleted records are removed. A new
container must load at least one valid record before serving traffic.

Pools use SHA-256 Rendezvous Hash over the secret name, session ID, and normalized
GitHub login. Token rotation and entry ordering do not change routing. Adding
an account moves some sessions to it. Removing an account moves only its sessions.
There is no session database or historical pool configuration. An upstream failure
never triggers a retry using another account. A rejected old session must be
abandoned and a new session started by the client.

For pools, generation requests must include an unambiguous `session-id`,
`session_id`, `thread-id`, or `x-claude-code-session-id`. Missing or conflicting
IDs receive HTTP 400. GET `/models` and `/v1/models` use deterministic discovery
routing. Single-account records do not require session headers.

Local tools use Azure CLI authentication and a nonsecret file at
`~/.config/claude-proxy/key-vault.json` with `vault_url` and `subscription`.
Install `copilot_vault.py` at `~/.local/lib/copilot-vault/copilot_vault.py`.
`cx` discovers records when invoked and updates local Codex provider keys when
switching. Codex still stores the selected client credentials locally.
`check_usage.py --json` reports every discovered account, deduplicating identical
login/token pairs across pools. Different credentials for one login are verified
separately. Vault failures are reported without falling back to local credentials.
