from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from typer.testing import CliRunner

from acsi.cert.build import build_certificate, canonical_payload_bytes
from acsi.cert.render import render_report
from acsi.cli import app
from acsi.importers.jsonl import import_jsonl_paths
from acsi.publish import publish_certificate
from acsi.sampling import sample_traces
from acsi.schemas import SamplingConfig, TraceRecord, WorkloadManifest
from acsi.scrub import scrub_traces

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
BANNED_RE = re.compile(
    r"\b(?:guarantee|guaranteed|identical|zero risk|proven equivalent)\b",
    re.IGNORECASE,
)
RAW_PII_RE = re.compile(
    r"ada@example\.com|ops@example\.com|lead@example\.com|312-555-0101|415-555-0102",
    re.IGNORECASE,
)


def test_spec_c_run_interrupt_mid_judge_resume_matches_control(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signing_key = _signing_key_b64()
    monkeypatch.setenv("ACSI_SIGNING_KEY", signing_key)
    run_id = "00000000-0000-0000-0000-0000000006c0"
    started_at = "2026-07-16T00:00:00Z"
    resume_root = tmp_path / "resume"
    control_root = tmp_path / "control"
    resume_manifest = _write_manifest(resume_root / "acsi.yaml", n=40, assertions=[])
    control_manifest = _write_manifest(control_root / "acsi.yaml", n=40, assertions=[])
    _preseed_run_json(resume_root, run_id, started_at)
    _preseed_run_json(control_root, run_id, started_at)

    interrupted = CliRunner().invoke(
        app,
        _run_args(
            resume_manifest,
            resume_root,
            run_id,
            extra=[
                "--inject-broken-json-rate",
                "0.10",
                "--interrupt-after-judge-dispatches",
                "3",
            ],
        ),
    )
    assert interrupted.exit_code == 1
    assert "Judge run interrupted" in interrupted.output

    resumed = CliRunner().invoke(
        app,
        _run_args(
            resume_manifest,
            resume_root,
            run_id,
            extra=["--inject-broken-json-rate", "0.10"],
        ),
    )
    control = CliRunner().invoke(
        app,
        _run_args(
            control_manifest,
            control_root,
            run_id,
            extra=["--inject-broken-json-rate", "0.10"],
        ),
    )
    assert resumed.exit_code == 0, resumed.output
    assert control.exit_code == 0, control.output

    resumed_cert = _cert(resume_root, run_id)
    control_cert = _cert(control_root, run_id)
    resumed_payload_sha = _payload_sha(resumed_cert)
    control_payload_sha = _payload_sha(control_cert)

    assert resumed_payload_sha == control_payload_sha
    assert resumed_cert["signature"] == control_cert["signature"]


def test_verdict_matrix_single_failed_criteria_and_degraded(tmp_path: Path) -> None:
    traces = _fixture_traces(20)
    manifest_path = _write_manifest(tmp_path / "acsi.yaml", n=20)
    manifest = WorkloadManifest.model_validate(json.loads(manifest_path.read_text()))

    cases = {
        "critical_assertions": {
            "assertions": [
                {
                    "assertion_id": "json-valid",
                    "baseline_passed": True,
                    "candidate_passed": False,
                    "pair_id": str(traces[0].trace_id),
                    "severity": "critical",
                    "trace_id": str(traces[0].trace_id),
                }
            ],
            "clusters": [],
            "judgments": [],
            "noise_upper": 0.0,
        },
        "candidate_regression_rate": {
            "assertions": [],
            "clusters": [],
            "judgments": _judgments(traces, worse_count=5),
            "noise_upper": 0.0,
        },
        "critical_cluster_share": {
            "assertions": [],
            "clusters": [
                {
                    "cluster_id": "cluster-0",
                    "description": "Critical cluster",
                    "member_count": 1,
                    "name": "Critical cluster",
                    "pair_ids": [str(traces[0].trace_id)],
                    "severity": "worse_critical",
                    "share_of_sampled": 0.02,
                }
            ],
            "judgments": [],
            "noise_upper": 0.0,
        },
    }

    observed: dict[str, list[str]] = {}
    for criterion_id, case in cases.items():
        run_dir = tmp_path / criterion_id / ".acsi" / "runs" / criterion_id
        _write_cert_artifacts(run_dir, traces, **case)
        result = build_certificate(
            manifest=manifest,
            traces=traces,
            run_dir=run_dir,
            manifest_path=manifest_path,
        )
        failed = [
            criterion["id"]
            for criterion in result.payload["criteria"]
            if criterion.get("passed") is False
        ]
        observed[criterion_id] = failed
        assert result.payload["verdict"] == "BLOCK"
        assert failed == [criterion_id]

    degraded_run_dir = tmp_path / "degraded" / ".acsi" / "runs" / "degraded"
    _write_cert_artifacts(
        degraded_run_dir,
        traces,
        assertions=[],
        clusters=[],
        judgments=[],
        noise_upper=None,
        degraded=True,
    )
    degraded = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=degraded_run_dir,
        manifest_path=manifest_path,
        degraded=True,
    )
    criterion_b = degraded.payload["criteria"][1]
    assert observed == {
        "critical_assertions": ["critical_assertions"],
        "candidate_regression_rate": ["candidate_regression_rate"],
        "critical_cluster_share": ["critical_cluster_share"],
    }
    assert degraded.payload["mode"] == "degraded"
    assert criterion_b["id"] == "candidate_regression_rate"
    assert criterion_b["passed"] is None
    assert "Noise floor unavailable (degraded mode)" in degraded.payload["coverage_sentence"]


def test_tamper_verify_outputs_verbatim(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "acsi.yaml", n=3)
    manifest = WorkloadManifest.model_validate(json.loads(manifest_path.read_text()))
    traces = _fixture_traces(3)
    run_dir = tmp_path / ".acsi" / "runs" / "tamper"
    _write_cert_artifacts(
        run_dir,
        traces,
        assertions=[],
        clusters=[],
        judgments=[],
        noise_upper=0.0,
    )
    build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )

    good = CliRunner().invoke(app, ["verify", str(run_dir / "cert.json")])
    assert good.exit_code == 0
    assert good.output == "Certificate signature verified.\n"

    cert_text = (run_dir / "cert.json").read_text(encoding="utf-8")
    (run_dir / "cert.json").write_text(
        cert_text.replace('"verdict":"PASS"', '"verdict":"FAIL"', 1),
        encoding="utf-8",
    )
    bad = CliRunner().invoke(app, ["verify", str(run_dir / "cert.json")])
    assert bad.exit_code == 1
    assert bad.output == "Error: Certificate signature verification failed.\n"


def test_sampling_dedup_strata_and_seed_hash() -> None:
    records = _fixture_traces(30)
    near_duplicates = [
        records[0].model_copy(
            update={
                "request": records[0].request.model_copy(
                    update={
                        "messages": [
                            records[0].request.messages[0].model_copy(
                                update={
                                    "content": (
                                        records[0].request.messages[0].content
                                        + f" duplicate variant {index}"
                                    )
                                }
                            )
                        ]
                    }
                )
            }
        )
        for index in range(10)
    ]
    result = sample_traces(
        near_duplicates,
        SamplingConfig(n=10, stratify_by=["template_id"], seed=11, k_baseline=2),
    )
    assert len(result.records) == 1
    assert result.report["dedup"]["collapsed_count"] == 9

    stratum_records = [
        _with_template(record, "A" if index < 10 else "B")
        for index, record in enumerate(records[:15])
    ]
    first = sample_traces(
        stratum_records,
        SamplingConfig(n=6, stratify_by=["template_id"], seed=5, k_baseline=2),
    )
    second = sample_traces(
        stratum_records,
        SamplingConfig(n=6, stratify_by=["template_id"], seed=5, k_baseline=2),
    )
    counts = {row["key"]: row["sampled"] for row in first.report["strata"]}
    assert counts == {"template_id=A": 4, "template_id=B": 2}
    assert first.sha256 == second.sha256


def test_scrub_counts_judge_visible_text_and_cert_exemplars(tmp_path: Path) -> None:
    trace = _fixture_traces(1)[0]
    prompt = (
        "Contact ada@example.com, ops@example.com, and lead@example.com. "
        "Call 312-555-0101 or 415-555-0102."
    )
    trace = trace.model_copy(
        update={
            "request": trace.request.model_copy(
                update={
                    "messages": [trace.request.messages[0].model_copy(update={"content": prompt})]
                }
            )
        }
    )
    scrubbed = scrub_traces([trace])
    judge_visible_text = scrubbed.records[0].request.messages[0].content
    assert scrubbed.report["counts"] == {"email": 3, "phone": 2}
    assert "[EMAIL_1]" in judge_visible_text
    assert "[PHONE_1]" in judge_visible_text
    assert RAW_PII_RE.search(judge_visible_text) is None

    manifest_path = _write_manifest(tmp_path / "acsi.yaml", n=1)
    manifest = WorkloadManifest.model_validate(json.loads(manifest_path.read_text()))
    run_dir = tmp_path / ".acsi" / "runs" / "scrub"
    _write_cert_artifacts(
        run_dir,
        scrubbed.records,
        assertions=[],
        clusters=[
            {
                "cluster_id": "cluster-0",
                "description": "Scrubbed exemplar",
                "member_count": 1,
                "name": "Scrubbed exemplar",
                "pair_ids": [str(scrubbed.records[0].trace_id)],
                "severity": "worse_minor",
                "share_of_sampled": 1.0,
            }
        ],
        judgments=[],
        noise_upper=0.0,
    )
    cert = build_certificate(
        manifest=manifest,
        traces=scrubbed.records,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    exemplar = cert.payload["clusters"][0]["exemplars"][0]
    assert "[EMAIL_1]" in exemplar
    assert "[PHONE_1]" in exemplar
    assert RAW_PII_RE.search(json.dumps(cert.payload)) is None


def test_output_length_inflation_reports_1_3x(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "acsi.yaml", n=3)
    manifest = WorkloadManifest.model_validate(json.loads(manifest_path.read_text()))
    traces = _fixture_traces(3)
    run_dir = tmp_path / ".acsi" / "runs" / "tokens"
    _write_cert_artifacts(
        run_dir,
        traces,
        assertions=[],
        clusters=[],
        judgments=[],
        noise_upper=0.0,
        baseline_output_tokens=10,
        candidate_output_tokens=13,
    )
    cert = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    assert cert.payload["cost_latency"]["output_length_inflation"] == 1.3


def test_publish_mocktransport_and_no_url_cli_error(tmp_path: Path) -> None:
    cert = {
        "payload": {
            "candidate_disagreement": {"rate": 0.08},
            "clusters": [
                {
                    "cluster_id": "cluster-0",
                    "count": 1,
                    "description": "Broken JSON",
                    "exemplars": ["[EMAIL_1]"],
                    "name": "Broken JSON",
                    "patch_diff": "--- patch",
                    "severity": "worse_critical",
                    "share_of_sampled": 0.08,
                }
            ],
            "cost_latency": {"output_length_inflation": 1.3},
            "coverage": {"n": 1},
            "criteria": [],
            "mode": "standard",
            "noise_floor": {"upper": 0.0},
            "run_id": "publish",
            "verdict": "BLOCK",
        }
    }
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(cert, sort_keys=True), encoding="utf-8")
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200, text="ok")

    publish_certificate(
        cert_path,
        url="https://publish.test/cert",
        transport=httpx.MockTransport(handler),
    )
    assert "exemplars" not in posted[-1]["clusters"][0]
    assert "patch_diff" not in posted[-1]["clusters"][0]

    publish_certificate(
        cert_path,
        url="https://publish.test/cert",
        include_examples=True,
        transport=httpx.MockTransport(handler),
    )
    assert posted[-1]["clusters"][0]["exemplars"] == ["[EMAIL_1]"]
    assert posted[-1]["clusters"][0]["patch_diff"] == "--- patch"

    no_url = CliRunner().invoke(app, ["publish", "--cert", str(cert_path)])
    assert no_url.exit_code == 1
    assert no_url.output == "Error: Pass --url or set ACSI_PUBLISH_URL to publish a certificate.\n"


def test_html_report_is_self_contained_static_and_complete(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "acsi.yaml", n=3)
    manifest = WorkloadManifest.model_validate(json.loads(manifest_path.read_text()))
    traces = _fixture_traces(3)
    run_dir = tmp_path / ".acsi" / "runs" / "report"
    _write_cert_artifacts(
        run_dir,
        traces,
        assertions=[],
        clusters=[
            {
                "cluster_id": "cluster-0",
                "description": "Cluster text",
                "member_count": 1,
                "name": "Cluster text",
                "pair_ids": [str(traces[0].trace_id)],
                "severity": "worse_minor",
                "share_of_sampled": 1 / 3,
            }
        ],
        judgments=[],
        noise_upper=0.0,
    )
    cert = build_certificate(
        manifest=manifest,
        traces=traces,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )
    render_report(cert.cert, output_path=run_dir / "report.html")
    html = (run_dir / "report.html").read_text(encoding="utf-8")

    assert "http://" not in html
    assert "https://" not in html
    assert "window.Alpine" in html
    # Section headings updated for the redesigned report (rendering-only change).
    for section in (
        "Pass criteria",
        "Failure clusters",
        "Noise floor",
        "Judge panel",
        "Cost",
        "Scope",
        "Verified:",
    ):
        assert section in html
    assert cert.payload["coverage_sentence"] in html
    assert "Cluster text" in html
    assert not BANNED_RE.search(html)


def _run_args(
    manifest: Path,
    root: Path,
    run_id: str,
    *,
    extra: list[str] | None = None,
) -> list[str]:
    return [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(root / ".acsi"),
        "--run-id",
        run_id,
        "--fake-noise",
        "0.05",
        "--yes",
        "--json",
        *(extra or []),
    ]


def _write_manifest(
    path: Path,
    *,
    n: int,
    assertions: list[dict[str, Any]] | None = None,
) -> Path:
    payload = {
        "assertions": assertions
        if assertions is not None
        else [{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/fake-judge"}],
            "min_judges": 1,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": n, "seed": 42, "stratify_by": ["template_id"]},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "support-ticket-summary",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_cert_artifacts(
    run_dir: Path,
    traces: list[TraceRecord],
    *,
    assertions: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    noise_upper: float | None,
    degraded: bool = False,
    baseline_output_tokens: int = 10,
    candidate_output_tokens: int = 10,
) -> None:
    (run_dir / "baseline").mkdir(parents=True)
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "patches").mkdir()
    _write_json(
        run_dir / "run.json",
        {
            "run_id": run_dir.name,
            "run_started_at": "2026-07-16T00:00:00Z",
            "sampled_trace_hash": {"algorithm": "sha256", "value": "sample"},
        },
    )
    _write_json(
        run_dir / "sampling_report.json",
        {
            "sampling_mode": "exhaustive",
            "strata": [{"available": len(traces), "key": "all", "sampled": len(traces)}],
        },
    )
    _write_json(run_dir / "scrub_report.json", {"counts": {}, "records": len(traces)})
    if degraded:
        noise_floor = {
            "degraded": True,
            "noise_floor": "unavailable",
            "threshold_source": "default_degraded",
        }
    else:
        noise_floor = {
            "beyond_noise_ci": {
                "confidence": 0.95,
                "lower": 0.0,
                "rate": 0.0,
                "upper": noise_upper,
            },
            "degraded": False,
        }
    _write_json(run_dir / "baseline" / "noise_floor.json", noise_floor)
    _write_json(run_dir / "clusters.json", {"clusters": clusters, "stats": {}})
    _write_json(run_dir / "patches" / "patch_report.json", {"patches": []})
    _write_json(
        run_dir / "judge_stats.json",
        {
            "ensemble": {"krippendorff_alpha": None, "raw_agreement_percent": None},
            "judges": {"openai/fake-judge": {}},
        },
    )
    _write_jsonl(run_dir / "judgments.jsonl", judgments)
    _write_jsonl(run_dir / "assertion_results.jsonl", assertions)
    _write_jsonl(
        run_dir / "baseline" / "responses.jsonl",
        [_response(str(trace.trace_id), baseline_output_tokens) for trace in traces],
    )
    _write_jsonl(
        run_dir / "candidate" / "responses.jsonl",
        [_response(str(trace.trace_id), candidate_output_tokens) for trace in traces],
    )


def _judgments(traces: list[TraceRecord], *, worse_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        rows.append(
            {
                "judge": "openai/fake-judge",
                "outcome": "worse_minor" if index < worse_count else "equivalent",
                "pair_id": str(trace.trace_id),
            }
        )
    return rows


def _response(trace_id: str, output_tokens: int) -> dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "model": "model",
        "response": {
            "finish_reason": "stop",
            "latency_ms": 100,
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


def _fixture_traces(count: int) -> list[TraceRecord]:
    return import_jsonl_paths([FIXTURE_PATH]).records[:count]


def _with_template(record: TraceRecord, template_id: str) -> TraceRecord:
    meta = record.meta.model_copy(update={"template_id": template_id})
    return record.model_copy(update={"meta": meta})


def _preseed_run_json(root: Path, run_id: str, started_at: str) -> None:
    run_dir = root / ".acsi" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run.json",
        {"run_id": run_id, "run_started_at": started_at, "stages": {}},
    )


def _cert(root: Path, run_id: str) -> dict[str, Any]:
    return json.loads((root / ".acsi" / "runs" / run_id / "cert.json").read_text(encoding="utf-8"))


def _payload_sha(cert: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(cert["payload"])).hexdigest()


def _signing_key_b64() -> str:
    private_key = Ed25519PrivateKey.generate()
    raw = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    return base64.b64encode(raw).decode("ascii")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
