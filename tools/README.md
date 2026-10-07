# Local Copilot tools

`cx` selects fixed local Codex providers. It does not synchronize with a cloud
configuration and has no `sync` command. Configure API keys in
`~/.config/claude-proxy/copilot-clients.json`:

```json
{
  "schema_version": 1,
  "entries": [
    {"number": 1, "api_key": "<first client key>"},
    {"number": 5, "api_key": "<pool client key>"}
  ]
}
```

Keep this file private. It contains only client API keys, not GitHub tokens.
The existing `copilot-accounts.json` format remains supported when this file is
absent. Switching providers preserves unrelated Codex settings and creates a
private backup. `cx status` and `cx official` work without local proxy keys.

The server's configuration and encryption are handled by the Rust binary.
See [configuration settings](../docs/encrypted-settings.md). Pool names remain
`copilot-1`, `copilot-2`, and so on. `session_hash` uses the pool name, session ID,
and normalized GitHub login. Token rotation and ordering do not change routing.
Membership changes move only affected sessions. Failed requests never switch
accounts. Multi-account generation requires a consistent `session-id`,
`session_id`, `thread-id`, or `x-claude-code-session-id` header.

The existing quota monitor (`check_usage.py`, `copilot_vault.py`, and their tests)
is intentionally unchanged. Its data source is separate from the server and
`cx`, and requires separate migration if its old source is unavailable.
