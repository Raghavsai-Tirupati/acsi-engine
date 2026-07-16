# OSS Issue Summary Benchmark

This benchmark workload is built from public closed GitHub issue text with per-item provenance URLs. It is not customer data and it is not production traffic. An ACSI certificate produced from this workload certifies this public benchmark corpus only.

## Rebuild

Set `GITHUB_TOKEN` to a GitHub token with public repository read access, then run:

```bash
python scripts/build_oss_issue_corpus.py --n 300
python scripts/generate_benchmark_traces.py --yes
```

Use `--fake` on `generate_benchmark_traces.py` for offline smoke tests with zero provider spend.

## Attribution

Each corpus row includes `source_repo`, `issue_number`, and `html_url`. Preserve those fields in derived benchmark artifacts so public issue provenance remains available.
