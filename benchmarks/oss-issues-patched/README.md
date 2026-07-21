# OSS Issue Summary Benchmark — patched variant

This is the **patched variant** of the [`oss-issues`](../oss-issues/README.md)
benchmark. It certifies the *corrected* migration: the same
Claude Opus 4.1 → Claude Sonnet 5 swap, but with a system prompt that spells out
the JSON contract's hard limits so the candidate stops overflowing the schema and
wrapping output in markdown fences. It exists to demonstrate the
**BLOCK → patch → PASS** arc — run it after the unpatched `oss-issues` benchmark
BLOCKs, to show the same swap passing once the prompt is fixed.

The only differences from the unpatched variant are the workload name
(`oss-issue-summary-patched`) and the patch paragraph appended to
`system_prompt.txt`. The corpus, schema (`summary.schema.json`), fabrication
rubric (`fabrication.txt`), models, sampling, assertions, and thresholds are all
identical, so the two certificates are directly comparable.

## The patch (v2)

The following paragraph is appended verbatim to the end of `system_prompt.txt`:

> Hard limits: keep the summary under 350 characters; the absolute maximum is 400. Keep affected_area under 50 characters; the absolute maximum is 60. A short module or component name is enough for affected_area. Include at most 5 action_items. Output the raw JSON object only - no markdown code fences, no text before or after it.

**Why v2.** The v1 patch — which stated only the absolute caps (400 / 60
characters) — already cut candidate regressions from **149 to 18** on this
workload. This iteration adds *headroom targets*: it asks the model to aim well
under the caps (350 / 50) rather than right at them, and clarifies that a short
module or component name suffices for `affected_area`, so borderline outputs stop
grazing the hard limits.

## Reuse the existing corpus — do not re-pull

This variant **reuses the corpus already built for `oss-issues`**
(`benchmarks/oss-issues/corpus.jsonl`). Do not re-fetch from GitHub; no
`GITHUB_TOKEN` is needed. Only the baseline traces are regenerated, because the
system prompt changed.

Run these from the repository root with `PYTHONPATH=.` exported. Steps 1 and 3 are
the two that spend money and call live providers; step 2 is local and needs no keys.

```bash
# 1. Generate baseline traces with the PATCHED prompt, reusing the existing corpus
#    (needs ANTHROPIC_API_KEY; add --fake for an offline, zero-spend smoke test)
uv run python scripts/generate_benchmark_traces.py \
  --corpus benchmarks/oss-issues/corpus.jsonl \
  --manifest benchmarks/oss-issues-patched/acsi.yaml \
  --system-prompt benchmarks/oss-issues-patched/system_prompt.txt \
  --output benchmarks/oss-issues-patched/traces.jsonl \
  --yes

# 2. Normalize and validate the traces (no keys)
uv run acsi import jsonl benchmarks/oss-issues-patched/traces.jsonl \
  --out .acsi/traces/oss-issue-summary-patched.jsonl

# 3. Full certification run (needs ANTHROPIC_API_KEY, OPENAI_API_KEY, paid GEMINI_API_KEY)
uv run acsi run --manifest benchmarks/oss-issues-patched/acsi.yaml \
  --traces .acsi/traces/oss-issue-summary-patched.jsonl --live --yes
```

Cost and time track the unpatched benchmark (~$17, ~90 min); a live run resumes
automatically if interrupted. See the [unpatched README](../oss-issues/README.md)
for the full sequence, the paid-tier Gemini note, and provenance details.
