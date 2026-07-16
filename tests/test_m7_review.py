from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import acsi.review as review_module
from acsi.cert.build import build_certificate, verify_certificate
from acsi.cert.render import render_report
from acsi.cli import app
from acsi.importers.jsonl import import_jsonl_paths
from acsi.overrides import append_override
from acsi.review import ReviewError, create_review_server
from acsi.schemas import WorkloadManifest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
RUN_ID = "00000000-0000-0000-0000-000000000701"


def test_review_server_binds_loopback_and_serves_report(tmp_path: Path) -> None:
    manifest_path, manifest, traces, run_root, active_run_dir = _write_run(tmp_path, n=3)
    result = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=active_run_dir,
        manifest_path=manifest_path,
    )
    render_report(result.cert, output_path=active_run_dir / "report.html")

    with pytest.raises(ReviewError, match="loopback"):
        create_review_server(
            run_id=RUN_ID,
            run_dir=run_root,
            manifest_path=manifest_path,
            host="0.0.0.0",
        )

    server = create_review_server(
        run_id=RUN_ID,
        run_dir=run_root,
        manifest_path=manifest_path,
        port=0,
    )
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        body = _request(server.server_address[1], "GET", "/")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert "Review Queue" in body
    assert "Unresolved Queue" in body
    assert "Promote Assertion" in body


def test_review_server_handles_keyboard_interrupt_cleanly(monkeypatch) -> None:
    class FakeServer:
        server_address = ("127.0.0.1", 4321)

        def __init__(self) -> None:
            self.closed = False
            self.shutdown_called = False

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.closed = True

    fake = FakeServer()

    def fake_factory(**_kwargs):
        return fake

    monkeypatch.setattr(review_module, "create_review_server", fake_factory)

    review_module.serve_review(run_id=RUN_ID)

    assert fake.shutdown_called is True
    assert fake.closed is True


def test_review_api_appends_overrides_without_mutating_judgments(tmp_path: Path) -> None:
    manifest_path, _manifest, traces, run_root, active_run_dir = _write_run(tmp_path, n=3)
    _write_jsonl(
        active_run_dir / "judgments.jsonl",
        [
            {
                "judge": "openai/judge-a",
                "outcome": "worse_minor",
                "pair_id": str(traces[0].trace_id),
            },
            {
                "judge": "openai/judge-b",
                "outcome": "equivalent",
                "pair_id": str(traces[0].trace_id),
            },
        ],
    )
    _write_jsonl(
        active_run_dir / "assertion_results.jsonl",
        [
            {
                "assertion_id": "critical-json",
                "baseline_passed": True,
                "candidate_passed": False,
                "pair_id": str(traces[1].trace_id),
                "severity": "critical",
            }
        ],
    )
    before_judgments = (active_run_dir / "judgments.jsonl").read_bytes()
    server = create_review_server(
        run_id=RUN_ID,
        run_dir=run_root,
        manifest_path=manifest_path,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        payload = json.dumps(
            {
                "note": "human reviewer accepted semantic equivalence",
                "pair_id": str(traces[0].trace_id),
                "to_outcome": "equivalent",
            }
        )
        override_body = _request(server.server_address[1], "POST", "/api/override", payload)
        assertion_body = _request(
            server.server_address[1],
            "POST",
            "/api/override",
            json.dumps({"pair_id": str(traces[1].trace_id), "to_outcome": "equivalent"}),
            expected_status=400,
        )
        run_body = _request(server.server_address[1], "GET", "/api/run")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert json.loads(override_body)["to_outcome"] == "equivalent"
    assert "Assertion-derived outcomes are not overridable." in assertion_body
    assert (active_run_dir / "judgments.jsonl").read_bytes() == before_judgments
    assert len((active_run_dir / "overrides.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(run_body)["unresolved_queue"] == [
        {"outcome": "unresolved", "pair_id": str(traces[0].trace_id)}
    ]


def test_review_api_promotes_assertion_with_backup(tmp_path: Path) -> None:
    manifest_path, _manifest, _traces, run_root, _active_run_dir = _write_run(tmp_path, n=3)
    server = create_review_server(
        run_id=RUN_ID,
        run_dir=run_root,
        manifest_path=manifest_path,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        body = _request(
            server.server_address[1],
            "POST",
            "/api/promote-assertion",
            json.dumps(
                {
                    "id": "contains-safe-next-step",
                    "params": {"value": "next_step"},
                    "severity": "major",
                    "type": "contains",
                }
            ),
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    payload = json.loads(body)
    assert payload["assertion"]["id"] == "contains-safe-next-step"
    assert Path(payload["backup_path"]).exists()
    assert "contains-safe-next-step" in manifest_path.read_text(encoding="utf-8")


def test_cert_regeneration_recomputes_overridden_verdict_and_regressed_pairs(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, traces, run_root, active_run_dir = _write_run(tmp_path, n=20)
    regressed = [str(traces[0].trace_id), str(traces[1].trace_id)]
    _write_jsonl(
        active_run_dir / "judgments.jsonl",
        [
            {"judge": "openai/judge-a", "outcome": "worse_critical", "pair_id": pair_id}
            for pair_id in regressed
        ],
    )
    _write_json(
        active_run_dir / "clusters.json",
        {
            "clusters": [
                {
                    "cluster_id": "cluster-critical",
                    "description": "critical judge cluster",
                    "member_count": 2,
                    "name": "Critical judge cluster",
                    "pair_ids": regressed,
                    "severity": "worse_critical",
                    "share_of_sampled": 0.1,
                }
            ],
            "stats": {},
        },
    )

    before = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=active_run_dir,
        manifest_path=manifest_path,
    )
    before_signature = before.cert["signature"]
    before_sha = before.cert_sha256
    assert before.payload["verdict"] == "BLOCK"
    assert before.payload["regressed_pairs"] == {
        "by_source": {"assertion": 0, "both": 0, "judge": 2},
        "count": 2,
        "rate": 0.1,
    }

    for pair_id in regressed:
        append_override(
            active_run_dir,
            pair_id=pair_id,
            from_outcome="worse_critical",
            to_outcome="equivalent",
            note="reviewer accepted equivalent output after inspection",
        )
    after = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=active_run_dir,
        manifest_path=manifest_path,
    )
    render_report(after.cert, output_path=active_run_dir / "report.html")

    assert after.payload["verdict"] == "PASS"
    assert after.payload["human_overrides"]["count"] == 2
    assert after.payload["regressed_pairs"] == {
        "by_source": {"assertion": 0, "both": 0, "judge": 0},
        "count": 0,
        "rate": 0.0,
    }
    assert after.cert["signature"] != before_signature
    assert after.cert_sha256 != before_sha
    assert verify_certificate(active_run_dir / "cert.json")["payload"]["verdict"] == "PASS"
    report = (active_run_dir / "report.html").read_text(encoding="utf-8")
    assert (
        "2 judge outcome(s) were overridden by human review; original judge output is preserved "
        "in the run record."
    ) in report

    cli = CliRunner().invoke(
        app,
        [
            "cert",
            "--run",
            RUN_ID,
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(run_root),
            "--json",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["verdict"] == "PASS"


def test_override_notes_are_sanitized_on_cert_ingestion(tmp_path: Path) -> None:
    manifest_path, manifest, traces, _run_root, active_run_dir = _write_run(tmp_path, n=3)
    pair_id = str(traces[0].trace_id)
    _write_jsonl(
        active_run_dir / "judgments.jsonl",
        [{"judge": "openai/judge-a", "outcome": "worse_minor", "pair_id": pair_id}],
    )
    append_override(
        active_run_dir,
        pair_id=pair_id,
        from_outcome="worse_minor",
        to_outcome="equivalent",
        note="guaranteed identical after review",
    )

    result = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=active_run_dir,
        manifest_path=manifest_path,
    )
    cert_text = (active_run_dir / "cert.json").read_text(encoding="utf-8")
    assert result.payload["banned_language_sanitization_count"] >= 2
    assert "guaranteed" not in cert_text
    assert "identical" not in cert_text
    assert "term removed" in cert_text


def _request(
    port: int,
    method: str,
    path: str,
    body: str | None = None,
    *,
    expected_status: int = 200,
) -> str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"content-type": "application/json"} if body is not None else {}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read().decode("utf-8")
    finally:
        connection.close()
    assert response.status == expected_status, payload
    return payload


def _write_run(
    tmp_path: Path,
    *,
    n: int,
) -> tuple[Path, WorkloadManifest, list, Path, Path]:
    manifest_path = tmp_path / "acsi.yaml"
    manifest_payload = {
        "assertions": [],
        "baseline": {"provider": "anthropic", "model": "claude-old"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-new"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/judge-a"}, {"model": "openai/judge-b"}],
            "min_judges": 2,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": n, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "demo",
    }
    _write_json(manifest_path, manifest_payload)
    manifest = WorkloadManifest.model_validate(manifest_payload)
    traces = import_jsonl_paths([FIXTURE_PATH]).records[:n]
    run_root = tmp_path / ".acsi"
    active_run_dir = run_root / "runs" / RUN_ID
    _write_artifacts(active_run_dir, traces)
    return manifest_path, manifest, traces, run_root, active_run_dir


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
    _write_json(run_dir / "scrub_report.json", {"counts": {}, "records": len(traces)})
    _write_json(
        run_dir / "sampling_report.json",
        {"sampling_mode": "exhaustive", "strata": []},
    )
    _write_json(
        run_dir / "baseline" / "noise_floor.json",
        {
            "beyond_noise_ci": {
                "confidence": 0.95,
                "lower": 0.0,
                "rate": 0.0,
                "upper": 1.0,
            },
            "degraded": False,
            "tau": 0.9,
            "threshold_source": "test",
        },
    )
    _write_json(run_dir / "clusters.json", {"clusters": [], "stats": {}})
    _write_json(run_dir / "patches" / "patch_report.json", {"patches": []})
    _write_json(
        run_dir / "judge_stats.json",
        {
            "ensemble": {"krippendorff_alpha": None, "raw_agreement_percent": None},
            "judges": {"openai/judge-a": {}, "openai/judge-b": {}},
            "run": {"completed_pairs": 0},
        },
    )
    _write_jsonl(run_dir / "judgments.jsonl", [])
    _write_jsonl(run_dir / "assertion_results.jsonl", [])
    _write_jsonl(
        run_dir / "baseline" / "responses.jsonl",
        [_response(str(trace.trace_id), "{}") for trace in traces],
    )
    _write_jsonl(
        run_dir / "candidate" / "responses.jsonl",
        [_response(str(trace.trace_id), "{}") for trace in traces],
    )
    _write_jsonl(
        run_dir / "sampled_traces.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )


def _response(trace_id: str, text: str) -> dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "model": "model",
        "response": {
            "finish_reason": "stop",
            "latency_ms": 1,
            "served_model": "model",
            "text": text,
            "tool_calls": None,
        },
        "retry_count": 0,
        "sample_index": 0,
        "served_model": "model",
        "status": "done",
        "trace_id": trace_id,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str))
            handle.write("\n")
