# Working Conventions

- Type hints everywhere.
- `ruff check` and `pytest` must pass before any milestone is called done.
- No TODO placeholders in completed milestones.
- Prefer stdlib.
- New dependencies require a one-line justification in the commit message.
- Never print or log trace content at INFO or above.
- Secrets only from environment variables.
- Small commits per module with imperative messages.
- When the spec is ambiguous, implement the simplest version that passes the acceptance tests and leave a `# SPEC-NOTE:` comment explaining the interpretation.

