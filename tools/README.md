# Local Copilot tools

`cx` reads providers directly from `~/.codex/config.toml` and changes only
`model_provider`. It preserves addresses, API keys, models, and other settings.
There is no sync command or separate key configuration file.

Use `cx copilot3` to select the existing `model_providers.copilot3` entry,
`cx official` for OpenAI, and `cx status` to inspect the current default.
Running `cx` in a terminal opens a menu of configured Copilot providers.
Switching creates a private backup and verifies the saved configuration.
New defaults apply after restarting Codex and starting a new chat.

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
