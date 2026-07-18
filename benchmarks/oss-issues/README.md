# OSS Issue Summary Benchmark

This benchmark workload is built from public closed GitHub issue text with per-item provenance URLs. It is not customer data and it is not production traffic. An ACSI certificate produced from this workload certifies this public benchmark corpus only.

## Rebuild

Set `GITHUB_TOKEN` to a GitHub token with public repository read access, then run:

```bash
python scripts/build_oss_issue_corpus.py --n 300
python scripts/generate_benchmark_traces.py --yes
```

Use `--fake` on `generate_benchmark_traces.py` for offline smoke tests with zero provider spend.

## Full run sequence

Run these steps from the repository root, in order. Each step lists the env vars
it needs; unset any that a step does not list.

1. **Preflight** — verify every provider credential and reachability before spending.
   Requires: `ANTHROPIC_API_KEY` (baseline + candidate), `OPENAI_API_KEY` and
   `GEMINI_API_KEY` (judge panel). `local` judges need no key.

   ```bash
   acsi preflight --manifest benchmarks/oss-issues/acsi.yaml
   ```

   Exits 0 when all keys are present and every model answers a 1-token probe;
   exits 1 and names any missing env var otherwise. Preflight cost is < $0.01.

2. **Build corpus** — fetch public closed GitHub issues into `corpus.jsonl`.
   Requires: `GITHUB_TOKEN` (public repo read).

   ```bash
   python scripts/build_oss_issue_corpus.py --n 300
   ```

3. **Generate traces** — replay the corpus against the baseline model into
   `traces.jsonl`. Requires: `ANTHROPIC_API_KEY`. Use `--fake` for an offline,
   zero-spend smoke test.

   ```bash
   python scripts/generate_benchmark_traces.py --yes
   ```

4. **Import** — normalize and validate the generated traces. Requires: no keys.

   ```bash
   acsi import jsonl benchmarks/oss-issues/traces.jsonl \
     --out .acsi/traces/oss-issue-summary.jsonl
   ```

5. **Run** — full certification pipeline (scrub → sample → baseline → replay →
   diff → judge → cluster → verdict → cert). Requires: `ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`, `GEMINI_API_KEY`. Pass `--live` to call the real providers;
   the preflight table then shows `Mode: LIVE` and the estimated spend, and a
   missing credential aborts before any provider call.

   ```bash
   acsi run --manifest benchmarks/oss-issues/acsi.yaml \
     --traces .acsi/traces/oss-issue-summary.jsonl --live --yes
   ```

   `acsi run` is **fake-by-default**: without `--live` it drives the whole
   pipeline with deterministic `Fake*` clients at zero provider spend. Fake mode
   is for wiring, CI, and dry-runs — its certificate is watermarked
   `client_mode: fake` and its `report.html` carries a "FAKE CLIENTS — NOT A
   CERTIFICATION" banner. Only a `--live` run produces a real certificate.

## Attribution

Each corpus row includes `source_repo`, `issue_number`, and `html_url`. Preserve those fields in derived benchmark artifacts so public issue provenance remains available.
