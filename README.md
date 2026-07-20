# ACSI — model-swap certifier

[![CI](https://github.com/Raghavsai-Tirupati/acsi-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Raghavsai-Tirupati/acsi-engine/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

ACSI replays your real LLM traffic against a candidate model, measures the baseline
model's own run-to-run variance first (the **noise floor**), checks your
customer-defined assertions, escalates the ambiguous pairs to cross-family blind
judges, and issues an **ed25519-signed PASS/BLOCK certificate** that anyone can
verify offline. It runs entirely in your environment — traces never leave by
default, and the only outbound calls are to the model providers you configure.

---

## The finding

On a strict-JSON summarization workload built from **279 real GitHub issues**
(kubernetes, vscode, react, rust), **Claude Sonnet 5 broke the output contract on
53% of inputs (149/279)** — verbosity blew the schema's length caps and wrapped
output in markdown fences — while **Claude Opus 4.1 broke 0**. On every pair that
resolved, blind **cross-family** judges rated Sonnet's content equal to or better
than Opus's: the swap fails on contract compliance, not on answer quality. The
result replicated at ~50% across three independent runs.

Scope this claim exactly as stated: **on this benchmark workload.** ACSI certifies
the corpus it was run against and nothing beyond it.

Signed certificate: **<https://www.acsi.dev/cert/opus-4-1-to-sonnet-5>**. Anyone can
download the raw certificate at
<https://www.acsi.dev/cert/opus-4-1-to-sonnet-5/cert.json> and check it offline with
`acsi verify cert.json` — no keys, no network, no trust in us required.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/Raghavsai-Tirupati/acsi-engine.git
cd acsi-engine
uv sync
export PYTHONPATH=.   # the acsi package is imported from the repo root
```

> **PYTHONPATH note:** running the `acsi` CLI from a source checkout requires
> `PYTHONPATH=.` (exported once per shell, as above). Without it the entry point
> fails with `ModuleNotFoundError: No module named 'acsi'`.

**Capture real traffic.** Wrap your existing LLM client with the fail-open
reference writer so each request/response is appended to a JSONL file — it never
raises into your app and drops on backpressure:

```python
from acsi.capture.python import AsyncJsonlWriter, capture_event

writer = AsyncJsonlWriter("traffic.jsonl")          # one line to start recording
capture_event(writer, {"prompt": prompt, "response": response, "source": "capture"})
```

Then normalize it: `uv run acsi import jsonl traffic.jsonl --out .acsi/traces/mine.jsonl`.

**Run the fake-mode demo first** — no network, no provider spend:

```bash
uv run acsi demo
```

This writes two self-contained runs under `.acsi/runs/` (a PASS and a BLOCK), each
with a `cert.json` and a `report.html`.

**Verify the signed certificate offline** — no keys, no network:

```bash
uv run acsi verify .acsi/runs/demo-pass/cert.json
# -> Certificate signature verified.
```

To certify your own traffic, point `acsi run` at your traces. It is
**fake-by-default** (zero spend, watermarked cert); add `--live --yes` to call the
real providers and produce a real certificate:

```bash
uv run acsi run --manifest acsi.yaml --traces .acsi/traces/mine.jsonl          # dry run
uv run acsi run --manifest acsi.yaml --traces .acsi/traces/mine.jsonl --live --yes
```

---

## Reproduce the benchmark

The OSS-issue benchmark is public GitHub issue text, not customer data. A full live
run costs roughly **~$17** and takes **~90 minutes**. Run from the repo root with
`PYTHONPATH=.` exported.

You need a **GitHub token** with public-repo read access (`GITHUB_TOKEN`) to build
the corpus, `ANTHROPIC_API_KEY` for baseline + candidate, and `OPENAI_API_KEY` plus
a **paid-tier** `GEMINI_API_KEY` for the judge panel (the Gemini free tier
rate-limits and drops judges, leaving pairs unresolved). Do not commit any token.

```bash
# 0. Preflight — verify every credential and reachability before spending (< $0.01)
uv run acsi preflight --manifest benchmarks/oss-issues/acsi.yaml

# 1. Build the corpus from public closed issues  (needs GITHUB_TOKEN)
uv run python scripts/build_oss_issue_corpus.py --n 300

# 2. Generate baseline traces  (needs ANTHROPIC_API_KEY; use --fake for a zero-spend smoke test)
uv run python scripts/generate_benchmark_traces.py --yes

# 3. Normalize and validate the traces  (no keys)
uv run acsi import jsonl benchmarks/oss-issues/traces.jsonl \
  --out .acsi/traces/oss-issue-summary.jsonl

# 4. Full certification run  (needs ANTHROPIC_API_KEY, OPENAI_API_KEY, paid GEMINI_API_KEY)
uv run acsi run --manifest benchmarks/oss-issues/acsi.yaml \
  --traces .acsi/traces/oss-issue-summary.jsonl --live --yes
```

A live run **resumes automatically** — if it is interrupted, re-run the same
command and ACSI reuses the banked baseline/candidate/judge calls instead of paying
for them again. See [benchmarks/oss-issues/README.md](benchmarks/oss-issues/README.md)
for the full sequence and provenance notes.

---

## Architecture

```
capture        record real traffic to JSONL (fail-open reference wrapper)
scrub          strip PII (regex by default; presidio via the [scrub] extra)
sample / dedup  stratified sampling; single-turn stateless workloads only
noise floor    measure the baseline model's own run-to-run variance first
candidate replay  run the same prompts through the candidate model
assertions     check customer-defined contracts by severity
similarity gate   only ambiguous pairs escalate past the deterministic diff
judge panel    cross-family judges, order-swapped to cancel position bias
clustering     group regressions with human-readable reasons + exemplars
three criteria  noise floor, assertions, and judged quality decide the verdict
signed cert    ed25519 PASS/BLOCK, verifiable offline against an embedded public key
```

Two design principles hold throughout: a **fail-closed evidence floor** — too few
valid judge verdicts resolves to `unresolved`, never a lucky PASS — and
**cross-invocation resume**, so an interrupted live run never re-pays for banked
provider calls.

---

## Status

v1, three weeks old, built by the founding team. Issues and pull requests welcome.

Licensed under [Apache-2.0](LICENSE).
