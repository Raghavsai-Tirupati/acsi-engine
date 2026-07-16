from __future__ import annotations

import json
from pathlib import Path

import pytest

from acsi.cert.build import (
    BannedLanguageError,
    CertificateVerificationError,
    build_certificate,
    verify_certificate,
)
from acsi.cert.render import render_report
from acsi.importers.jsonl import import_jsonl_paths
from acsi.schemas import WorkloadManifest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
RUN_ID = "00000000-0000-0000-0000-000000000601"


def test_build_certificate_signs_renders_and_verifies(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)

    result = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    report_hash = render_report(result.cert, output_path=run_dir / "report.html")

    assert result.payload["verdict"] == "PASS"
    assert result.payload["criteria"][0]["passed"] is True
    assert "0 critical failures observed at n=3" in result.payload["coverage"][
        "zero_event_bound_sentence"
    ]
    assert result.payload["cost_latency"]["tokenizer_inflation"] == 2.0
    assert verify_certificate(run_dir / "cert.json")["payload"]["verdict"] == "PASS"
    assert report_hash
    assert b"\r\n" not in (run_dir / "cert.json").read_bytes()


def test_verify_fails_when_payload_is_tampered(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)
    build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    cert_path = run_dir / "cert.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    cert["payload"]["verdict"] = "BLOCK"
    cert_path.write_text(
        json.dumps(cert, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CertificateVerificationError):
        verify_certificate(cert_path)


def test_build_certificate_rejects_authored_banned_language(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)

    with pytest.raises(BannedLanguageError):
        build_certificate(
            manifest=manifest,
            traces=traces,
            run_dir=run_dir,
            manifest_path=manifest_path,
            authored_context=["This is guaranteed."],
        )


def test_render_rejects_doctored_template(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)
    result = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "alpine.min.js").write_text("window.Alpine={start(){}};", encoding="utf-8")
    (template_dir / "report.html.j2").write_text(
        "<html><body>guaranteed {{ payload.verdict }}</body></html>",
        encoding="utf-8",
    )

    with pytest.raises(BannedLanguageError):
        render_report(
            result.cert,
            output_path=run_dir / "bad.html",
            template_dir=template_dir,
        )


def test_model_generated_banned_language_is_sanitized(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)
    _write_json(
        run_dir / "clusters.json",
        {
            "clusters": [
                {
                    "cluster_id": "cluster-0",
                    "description": "guaranteed identical output",
                    "member_count": 1,
                    "name": "proven equivalent",
                    "pair_ids": [str(traces[0].trace_id)],
                    "severity": "worse_minor",
                    "share_of_sampled": 1 / 3,
                }
            ],
            "stats": {},
        },
    )

    result = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    render_report(result.cert, output_path=run_dir / "report.html")

    cert_text = (run_dir / "cert.json").read_text(encoding="utf-8")
    report_text = (run_dir / "report.html").read_text(encoding="utf-8")
    assert result.payload["banned_language_sanitization_count"] >= 2
    assert "term removed" in cert_text
    assert "guaranteed" not in cert_text
    assert "proven equivalent" not in report_text


def _write_inputs(tmp_path: Path) -> tuple[Path, WorkloadManifest, list, Path]:
    manifest_path = tmp_path / "acsi.yaml"
    manifest_payload = {
        "assertions": [],
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/fake-judge"}],
            "min_judges": 1,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 3, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "volunteer-application-summary",
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = WorkloadManifest.model_validate(manifest_payload)
    traces = import_jsonl_paths([FIXTURE_PATH]).records[:3]
    run_dir = tmp_path / ".acsi" / "runs" / RUN_ID
    _write_artifacts(run_dir, traces)
    return manifest_path, manifest, traces, run_dir


def _write_artifacts(run_dir: Path, traces: list) -> None:
    (run_dir / "baseline").mkdir(parents=True)
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "patches").mkdir()
    _write_json(
        run_dir / "run.json",
        {
            "run_id": RUN_ID,
            "run_started_at": "2026-07-16T00:00:00Z",
            "sampled_trace_hash": {"algorithm": "sha256", "value": "abc"},
        },
    )
    _write_json(
        run_dir / "sampling_report.json",
        {
            "sampling_mode": "exhaustive",
            "strata": [{"available": 3, "key": "all", "sampled": 3}],
        },
    )
    _write_json(run_dir / "scrub_report.json", {"counts": {}, "records": 3})
    _write_json(
        run_dir / "baseline" / "noise_floor.json",
        {
            "beyond_noise_ci": {
                "confidence": 0.95,
                "lower": 0.0,
                "rate": 0.0,
                "upper": 0.0,
            },
            "degraded": False,
        },
    )
    _write_json(run_dir / "clusters.json", {"clusters": [], "stats": {}})
    _write_json(run_dir / "patches" / "patch_report.json", {"patches": []})
    _write_json(
        run_dir / "judge_stats.json",
        {
            "ensemble": {"krippendorff_alpha": None, "raw_agreement_percent": None},
            "judges": {"openai/fake-judge": {}},
        },
    )
    _write_jsonl(run_dir / "judgments.jsonl", [])
    _write_jsonl(run_dir / "assertion_results.jsonl", [])
    _write_jsonl(
        run_dir / "baseline" / "responses.jsonl",
        [_response(str(trace.trace_id), 10, 100) for trace in traces],
    )
    _write_jsonl(
        run_dir / "candidate" / "responses.jsonl",
        [_response(str(trace.trace_id), 20, 125) for trace in traces],
    )


def _response(trace_id: str, output_tokens: int, latency_ms: int) -> dict:
    return {
        "cost_usd": 0.0,
        "model": "model",
        "response": {
            "finish_reason": "stop",
            "latency_ms": latency_ms,
            "served_model": "model",
            "text": "{}",
            "tool_calls": None,
        },
        "retry_count": 0,
        "sample_index": 0,
        "served_model": "model",
        "status": "done",
        "trace_id": trace_id,
        "usage": {"input_tokens": 1, "output_tokens": output_tokens},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
