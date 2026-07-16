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

# Standing Policies

- **Cross-platform:** path assertions compare `pathlib.Path` objects, never strings; every `open()` specifies `encoding="utf-8"`; any artifact whose hash is recorded is written with `"\n"` newlines explicitly; CI runs on ubuntu, macos, and windows and all three must pass.
- **SQLite:** `sqlite3.Connection` context managers handle transactions, NOT closing — always close via `contextlib.closing` or an explicit `close`; never rely on GC before touching or deleting a db file (Windows locks open files).
- **Reports:** every completion report enumerates each acceptance item individually with observed values (numbers, hashes, verbatim messages) — never summarized coverage claims.
- **SPEC-NOTE rule:** any interpretation call on pinned language, frozen contracts, or spec ambiguity gets a `# SPEC-NOTE:` comment in code AND a line in the report. Silent contract changes are defects.
- **Test adjustments:** existing test expectations may only change with a one-line justification in the commit message; no behavioral test may be weakened to make a symptom disappear.
- **User-facing strings never reference milestone numbers** (guard test exists).
- **No live network calls in tests or CI;** all tests use the `Fake*` clients; live paths are gated on env keys with guard tests.
- **Division of labor:** the coding agent writes code, runs local gates (`uv run ruff check .`, `uv run pytest`), commits per module with imperative messages, and pushes. It does NOT query GitHub Actions — CI verification is the operator's job.

