# ACSI Engine — Claude Code Build Prompt

You are building **ACSI v1** from scratch: a model swap certifier. It replays a company's real production LLM traffic against a candidate model, diffs every response with a noise-floor-calibrated methodology, and returns a signed PASS/BLOCK certificate with the exact failing prompts and proposed patches. Nothing exists yet — you are creating the repository, the package, the tests, and the CI from an empty directory. Do not assume any prior code exists.

This document is the complete spec. Follow it milestone by milestone. Where it is silent, prefer the simplest implementation that satisfies the acceptance tests.

---

## 1. Product context you must not violate

- **Problem:** AI providers force model changes several times a year (deprecations, successor upgrades, silent drift) and teams cannot test what breaks.
- **Solution one-liner:** ACSI replays your real traffic against every model change and certifies it before your customers find out.
- **Trust posture:** the engine runs entirely in the customer's environment. Traces never leave by default. The only outbound calls are to model providers the customer configures, plus an optional explicit `acsi publish` of the verdict JSON (never trace content unless `--include-examples`).
- **Verdict language is legally load-bearing.** Certificates state PASS or BLOCK "against your assertions at stated coverage and confidence." The words *guarantee*, *guaranteed*, *identical*, *zero risk*, and *proven equivalent* must never appear in any rendered certificate. You will write a unit test that fails the build if the renderer emits any of them.
- v1 scope: **single-turn, stateless workloads only** (classification, extraction, summarization, RAG answers, single-turn tool calls). Multi-turn is out of scope and must be excluded by the validator, with the exclusion percentage reported.

First real customer (dogfood): a healthcare-workforce app whose production workload is Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) summarizing volunteer applications, migrating to `claude-sonnet-5`. Build nothing specific to that domain beyond the fixtures — but the Supabase importer and the Anthropic→Anthropic judging rules below exist because of it.

## 2. Environment and stack

- Python **3.12+**, managed with **uv**. Package name `acsi`, CLI entry point `acsi`.
- Core dependencies (justify any addition beyond these in the commit message): `typer` (CLI), `pydantic` v2 (schemas), `litellm` (all provider calls), `httpx`, `rich` (progress/output), `numpy`, `scipy`, `scikit-learn` (clustering), `sentence-transformers` (lazy-imported; default model `BAAI/bge-small-en-v1.5`), `jinja2` (report), `cryptography` (ed25519 signing), `pytest`, `ruff`.
- Optional extras: `acsi[scrub]` → `presidio-analyzer` (heavy spaCy deps). A built-in regex scrubber (emails, phones, SSN-like, simple name patterns) must work without the extra.
- Persistence: **SQLite** (stdlib `sqlite3`) for run state, checkpoints, and response cache. JSONL for trace artifacts. No server databases.
- Secrets: environment variables only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`). All optional in development — every pipeline stage must run end-to-end with the FakeClient (§8) and shipped fixtures, spending zero dollars.
- Never log prompt or response content at INFO level or above. `--redact-logs` (default on in CI mode) hashes content in DEBUG logs too.
- Cross-platform from day one: `pathlib` everywhere, UTF-8 explicit, CI matrix on ubuntu + macos + windows.

## 3. Repository layout to create

```
acsi/
  pyproject.toml            # uv-managed; [project.scripts] acsi = "acsi.cli:app"
  CLAUDE.md                 # generated in M0 from §12 of this spec
  SPEC.md                   # this document, committed verbatim
  acsi/
    cli.py                  # typer app: init|import|run|baseline|replay|judge|cert|review|monitor|verify|publish
    schemas.py              # pydantic models; JSON Schema export command
    importers/jsonl.py
    importers/supabase.py
    importers/langfuse.py   # stub in v1: raise NotImplementedError with helpful message
    capture/python.py       # reference wrapper around anthropic/openai clients → JSONL
    capture/typescript.md   # documented TS snippet (see §7.1) — docs, not executed code
    scrub.py
    sampling.py
    replay/runner.py
    replay/params.py
    replay/clients.py       # CompletionClient protocol: LiveClient (litellm) + FakeClient
    replay/store.py         # SQLite checkpoint/cache/cost ledger
    diff/deterministic.py
    diff/semantic.py
    diff/clustering.py
    judge/rubric.py
    judge/ensemble.py
    judge/calibration.py
    stats.py
    patch.py
    cert/build.py
    cert/render.py          # Jinja → self-contained report.html (embedded JSON + Alpine.js)
    monitor.py
    review.py               # stdlib http.server or uvicorn serving report + override writeback
  templates/report.html.j2
  tests/                    # mirrors package; fixtures/ holds generated synthetic traces
  .github/workflows/ci.yml  # ruff + pytest, 3-OS matrix
```

## 4. Data contracts — freeze in M0, everything else builds on these

**TraceRecord** (one JSONL line per LLM call):

```json
{
  "trace_id": "uuid", "ts": "2026-07-15T18:22:03Z",
  "source": "capture|backfill|jsonl|supabase|langfuse",
  "workload": "kanban-summary",
  "request": {
    "provider": "anthropic", "model": "claude-haiku-4-5-20251001",
    "system": "...", "messages": [{"role": "user", "content": "..."}],
    "tools": null,
    "params": {"temperature": 0.2, "max_tokens": 512}
  },
  "response": {
    "text": "...", "tool_calls": null, "finish_reason": "end_turn",
    "usage": {"input_tokens": 1180, "output_tokens": 243}, "latency_ms": 2110,
    "served_model": "claude-haiku-4-5-20251001"
  },
  "meta": {"tags": ["prod"], "pii_scrubbed": false, "template_id": null}
}
```

Validator rules: exactly one user message; `system` optional; `response` may be empty only when `source: backfill`. Multi-turn records are rejected into an exclusions list, never silently dropped.

**WorkloadManifest** (`acsi.yaml` in the customer repo):

```yaml
workload: kanban-summary
baseline: {provider: anthropic, model: claude-haiku-4-5-20251001}
candidate: {provider: anthropic, model: claude-sonnet-5}
sampling: {n: 1000, stratify_by: [template_id, input_length_bucket], seed: 42, k_baseline: 2}
assertions:
  - {id: json-valid, type: json_schema, schema_ref: summary.schema.json, severity: critical}
  - {id: length, type: length_range, min_chars: 200, max_chars: 1200, severity: minor}
  - {id: latency, type: latency_p95_ms, max: 6000, severity: major}
  - {id: no-fabrication, type: judge_classifier, prompt_ref: fabrication.txt, severity: critical}
judging: {families_allowed: [openai, google, local], min_judges: 2}
thresholds: {epsilon_pp: 2.0, max_critical: 0, confidence: 0.95}
privacy: {scrub: true, egress: hosted_api}     # local | in_tenant | hosted_api
budget: {max_usd: 50, use_batch_api: false}    # batch path is post-v1; keep the flag
```

Assertion types to implement: `contains`, `not_contains`, `regex`, `json_schema`, `json_valid`, `numeric_field_equal`, `length_range`, `latency_p95_ms`, `refusal` (detector counts as critical fail when candidate refuses and baseline did not), `judge_classifier` (routed through the judge layer). Each has `severity: critical|major|minor`.

**RunManifest** (`run.json`, immutable): run_id; SHA-256 content hashes of the manifest and the sampled trace set; engine version; seeds; per-provider endpoint + every `served_model` string echoed from responses; every param transformation applied (§7.2); wall-clock; cost ledger (per stage, per provider, tokens in/out, USD).

**Certificate** (`cert.json`, ed25519-signed): verdict; scope + exclusion %; coverage (n, sampling method, strata, zero-event bound sentence); noise floor mean + 95% CI; candidate disagreement + CI + delta; assertion results by severity; judge panel (models, families, order-swap on, Krippendorff's α, % agreement, human-calibration accuracy if a calibration set was provided); regression clusters (name, count, redacted exemplars, patch diff if validated); cost + latency deltas including a tokenizer-inflation line; config hash, engine version, signature. `acsi verify cert.json` validates the signature offline against a public key embedded in the file's header block.

## 5. Pipeline behavior (what each stage must do)

1. **Import** (`acsi import jsonl|supabase`): normalize to TraceRecords, run the validator, print inventory summary (count, templates detected, exclusions). Supabase importer reads this table:
   ```sql
   create table acsi_traces (
     id uuid primary key default gen_random_uuid(),
     ts timestamptz not null default now(),
     workload text not null,
     request jsonb not null,
     response jsonb not null,
     meta jsonb not null default '{}'::jsonb
   );
   ```
2. **Scrub**: regex scrubber core, Presidio when the extra is installed; emits a scrub report (entity counts by type); sets `meta.pii_scrubbed`. Runs automatically before any content is sent to a judge whose provider family differs from both baseline and candidate.
3. **Sample**: stratified by `stratify_by` weighted by production frequency; minhash dedup of near-identical prompts; seeded; writes the sampled set + its content hash.
4. **Baseline / noise floor** (`acsi baseline`): replay the *baseline* model `k_baseline` times per sampled prompt at the trace's own params. Per-prompt self-agreement uses the same diff ladder as the candidate comparison. Output: noise-floor distribution + bootstrap CI, and the calibrated semantic threshold (§5.6). If the baseline model errors as retired/unavailable, switch to degraded mode: use stored historical responses as a single baseline sample and mark the run `noise_floor: unavailable` — verdict language weakens accordingly.
5. **Replay** (`acsi replay`): bounded concurrency, per-provider token bucket, exponential backoff on 429/5xx, SQLite checkpoint after every batch (resume, never restart), response cache keyed on (model, params-hash, prompt-hash, sample-index), running cost ticker, pre-run cost estimate, hard stop at `budget.max_usd` with a resumable checkpoint.
6. **Diff ladder** — each tier only sees what the previous couldn't settle:
   - Tier 1 deterministic: assertion engine + exact/normalized match, JSON parse + schema, numeric equality, refusal detector, length/latency.
   - Tier 2 semantic: local embedding cosine similarity vs a threshold **calibrated from this run's own noise floor** (5th percentile of baseline self-similarity). Never a hardcoded constant.
   - Tier 3 judges: only the borderline band (target 10–20% of pairs).
7. **Judges**: pairwise rubric — "given this prompt, does the candidate response satisfy the same task as the baseline: equivalent / better / worse-minor / worse-critical" — strict JSON output with a schema-validated parse and one retry on parse failure. Every pair judged twice with response order swapped; votes averaged. Panel selection: exclude the provider families of both baseline and candidate; require `min_judges` from `families_allowed`; `local` means an Ollama endpoint (OpenAI-compatible URL via litellm). Compute Krippendorff's α and % agreement per run. `judge/calibration.py` ingests a human-labeled CSV (pair_id, human_label) and reports judge-vs-human accuracy.
8. **Cluster + patch**: embed failing pairs, HDBSCAN, one judge call per cluster to name it. Patch proposer: detect a stable shared template (common prefix/suffix across sampled prompts); if none, skip patching and record why. If found, propose a minimal template-level edit as a unified diff, re-replay the failed subset plus a same-size clean control subset; attach the patch to the cert only if it fixes the cluster without regressing the control.
9. **Verdict + cert** (`acsi cert`): PASS requires zero critical assertion failures AND candidate-disagreement CI upper bound ≤ noise-floor CI upper bound + `epsilon_pp` AND no worse-critical cluster > 1% of samples. Render `cert.json` (signed) + self-contained `report.html`. Canonical coverage sentence: "{VERDICT} at n={n}, covering {pct}% of production template distribution, 95% CI {ci}. This certifies the sampled workload against the stated assertions; it does not certify unsampled inputs."

SPEC-NOTE: In M6, candidate disagreement is operationalized as the sampled-pair rate whose ensemble outcome is `worse_minor`, `worse_critical`, or `unresolved`; unresolved counts against PASS under the conservative M4 policy. Certificate output also applies the SPEC test E wording rule in two tiers: authored cert/template strings raise at build time, while ingested model-generated text is sanitized and counted. The original canonical sentence used the banned term "guarantee"; M6.5 resolves that upstream contradiction by pinning the banned-language-safe sentence above.
10. **Review** (`acsi review`): serve `report.html` on localhost; cluster browser, side-by-side diffs, judge-label override writeback (overrides recorded and footnoted on regenerated certs), "promote example to assertion" appends to `acsi.yaml`.
11. **Monitor** (`acsi monitor`): replay the golden suite (top strata + all assertion-bearing prompts, ~100–200) against the pinned production model on demand/cron; same noise-floor test; nonzero exit code + JSON summary on drift (CI-friendly).
12. **`acsi run`** = import-check → scrub → sample → baseline → replay → diff → judge → cluster → verdict → cert, with per-stage resume.

SPEC-NOTE: `acsi cluster --run <run_id>` is also exposed as a standalone stage command for composability, mirroring the separately resumable `baseline`, `replay`, and `judge` commands while preserving `acsi run` as the full pipeline entrypoint.

## 6. Critical engineering constraints (each of these has burned someone; do not skip)

1. **Capture is fail-open.** The reference wrappers (Python module + documented TypeScript snippet) must swallow and locally log their own errors, never raise into the host app, and write asynchronously with drop-on-backpressure. A certifier that breaks customer prod on day one is dead.
2. **Param mapping is mandatory** (`replay/params.py`). Verified July 2026: Anthropic returns HTTP 400 for non-default `temperature`/`top_p`/`top_k` on Claude Sonnet 5 and Opus 4.7+. A naive replay of Haiku traces against Sonnet 5 fails on request one. Implement a per-target transformation table (strip/translate/clamp), record every transformation in the RunManifest, and surface them on the cert — a param change is itself a disclosed behavioral variable. Design the table to be data-driven so new provider quirks are one-line additions.
3. **Truncation is not regression.** If the candidate hits `max_tokens` (tokenizer differences make this common — Sonnet 5 emits up to ~1.35× the tokens of the 4.6 family for the same text), flag it as a separate `truncated` class, normalize the output budget, and report it distinctly. Tokenizer-driven cost inflation gets its own line in the cert's cost-delta section.
4. **Pin what you can, record what you can't.** Request dated snapshot IDs where providers offer them; always store the `served_model` echoed in responses.
5. **Reproducibility:** all randomness seeded from the manifest; content hashes in the RunManifest; a re-run against the cache must be byte-identical.
6. **The banned-words test** (§1) is a real unit test against the rendered HTML and JSON, not a code-review convention.
7. Single-turn tool calls are in scope: compare tool name + arguments structurally (canonicalized JSON diff), which is Tier-1 deterministic.

## 7. Statistics requirements (`stats.py`)

Paired design (candidate vs baseline on the same prompt). Percentile bootstrap CIs, B=2,000, on disagreement rates and their delta. McNemar for binary assertion flips. Zero-event bound: with n samples and zero observed criticals, report the rule-of-three 95% upper bound ≈ 3/n in exactly that framing ("≤0.3% at n=1,000"), never "no criticals found." Keep `k_baseline` ≥ 2 even at temperature 0 — provider nondeterminism is measured, not assumed away. Benjamini–Hochberg note when >5 assertions are evaluated. Every statistical function gets a unit test against a hand-computed example.

## 8. Testing strategy — the FakeClient is the heart of it

Implement `CompletionClient` as a protocol with two implementations:
- `LiveClient`: litellm, used only when keys are present.
- `FakeClient`: deterministic canned responses keyed by prompt hash, with **injectable nondeterminism** (`noise=0.0–1.0` swaps in paraphrase variants at that rate) and **injectable regressions** (a rule like "when prompt contains X, return a broken-JSON/wrong-format response"). This is not just a mock — tunable noise is precisely what lets you unit-test the noise-floor statistics, threshold calibration, and verdict logic offline for free.

Fixtures: a generator script producing ~300 synthetic volunteer-application-style traces (fabricated names/data, varied lengths, one shared template + 10% templateless strays to exercise the patch guard).

End-to-end acceptance tests (these define done):
- **A. Clean pass:** FakeClient, noise=0.05 both models, no injected regression → `acsi run` produces a signed PASS cert; noise floor CI overlaps candidate disagreement; zero-event sentence present; `acsi verify` passes.
- **B. Caught regression:** candidate FakeClient injects broken JSON on 8% of prompts matching a token → BLOCK; the JSON-schema assertion shows critical failures; a cluster is named; the patch pathway triggers on the templated subset.
- **C. Resume:** kill the replay mid-run (simulate), rerun, zero duplicate spend (cache hit count asserted), identical final cert hash.
- **D. Param gotcha:** trace with `temperature: 0.2` replayed against a target whose mapping table strips it → transformation recorded in RunManifest and rendered on cert.
- **E. Banned words:** renderer fed a doctored context attempting to include "guaranteed" → build fails.

## 9. Build order — work strictly in milestones; finish, test, and demo each before the next

- **M0 Scaffold:** uv project, layout from §3, pydantic schemas + `acsi schema export`, fixtures generator, ruff + pytest + 3-OS CI, CLAUDE.md generated from §12, typer stubs for every command (helpful "not implemented" messages). *Accept: CI green, `uvx --from . acsi --help` shows all commands.*
- **M1 Import:** jsonl + supabase importers, validator, inventory summary. *Accept: fixtures import with correct exclusion counts.*
- **M2 Replay core:** clients (Fake + Live), params.py, runner with checkpoint/cache/budget/token bucket, cost ledger. *Accept: acceptance test C + D pass.*
- **M3 Diff + stats:** deterministic tier, assertion engine, semantic tier with calibrated threshold, noise-floor pipeline, stats.py with unit tests. *Accept: noise floor on FakeClient(noise=0.05) lands near 5% with sane CI.*
- **M4 Judges:** rubric, order-swap, family exclusion, ensemble stats, calibration ingest, FakeJudge. *Accept: α computed; family-exclusion unit test (anthropic→anthropic run refuses an anthropic judge).*
- **M5 Cluster + patch:** HDBSCAN, cluster naming, template detection guard, patch propose/validate loop. *Accept: acceptance test B names the injected cluster.*
- **M6 Verdict + cert:** thresholds, cert build + signing, HTML report, `acsi verify`, banned-words test. *Accept: tests A, B, E pass end-to-end.*
- **M7 Review + monitor:** local review server with overrides, monitor with CI exit codes. *Accept: override changes a judge label and the regenerated cert footnotes it; monitor exits nonzero on injected drift.*

## 10. Out of scope — do not build even if it seems easy

Multi-turn/agent replay, model routing, provider Batch API path (flag exists, path raises NotImplementedError), Langfuse/LangSmith importers beyond stubs, warranty language, auth/accounts, billing, web dashboard (the Next.js cert page is a separate repo — this engine only emits `cert.json` and POSTs it when `acsi publish` is explicitly called), desktop GUI, fine-tuning, SOC 2 tooling.

## 11. CLI UX standards

`rich` progress with a live cost ticker during replay; every command supports `--json` for machine-readable output; errors are actionable sentences ("Baseline model returned 404 — it may be retired. Rerun with --degraded to certify against stored outputs."); `acsi run` prints a pre-flight table (traces, n, estimated cost, estimated wall-clock, providers touched) and requires `--yes` or interactive confirm before spending.

## 12. Working conventions (also becomes CLAUDE.md)

Type hints everywhere; `ruff check` and `pytest` must pass before any milestone is called done; no TODO placeholders in completed milestones; prefer stdlib; new dependencies require a one-line justification in the commit message; never print or log trace content at INFO+; secrets only from env; small commits per module with imperative messages; when the spec is ambiguous, implement the simplest version that passes the acceptance tests and leave a `# SPEC-NOTE:` comment explaining the interpretation.

## 13. First actions, in order

1. `uv init --package acsi && cd acsi` — set Python 3.12, add core deps from §2.
2. Commit this document as `SPEC.md`; generate `CLAUDE.md` from §12.
3. Build M0. Stop at the end of each milestone, run the full test suite, and summarize what was built, what was deferred, and any SPEC-NOTEs before continuing.
