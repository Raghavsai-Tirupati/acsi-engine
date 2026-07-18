from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

import acsi.cli as cli
from acsi.cert.build import EvidenceFloorError, build_certificate
from acsi.cli import app
from acsi.replay.clients import CompletionResponse, map_litellm_error

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"


def _write_manifest(path: Path) -> Path:
    payload = {
        "assertions": [{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "budget": {"max_usd": 1000.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/fake-judge"}],
            "min_judges": 1,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 300, "seed": 42, "stratify_by": ["template_id"]},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "support-ticket-summary",
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _base_args(manifest: Path, run_dir: Path, run_id: str) -> list[str]:
    return [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]


def test_a_healthy_run_covers_evidence_and_certifies(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    result = CliRunner().invoke(app, _base_args(manifest, run_dir, "ok"))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    active = Path(payload["run_dir"])

    sampled = _count_lines(active / "sampled_traces.jsonl")
    candidate = _count_lines(active / "candidate" / "responses.jsonl")
    # The live-seam invariant the fail-open run violated: candidate rows exist and
    # cover every sampled pair BEFORE a verdict is issued.
    assert candidate > 0
    assert candidate == sampled

    cert = json.loads((active / "cert.json").read_text(encoding="utf-8"))["payload"]
    assert cert["coverage"]["n"] == sampled
    assert cert["verdict"] in {"PASS", "BLOCK"}
    # Broken-json injection guarantees critical assertion failures were evaluated.
    assert _count_lines(active / "assertion_results.jsonl") > 0


def test_b_billing_rejection_midway_candidate_aborts_named_error_no_cert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BillingMidCandidate:
        candidate_model = "claude-sonnet-5"
        ok_candidate = 3
        _seen = 0
        _lock = threading.Lock()

        def complete(self, request):
            if request.model == BillingMidCandidate.candidate_model:
                with BillingMidCandidate._lock:
                    BillingMidCandidate._seen += 1
                    seen = BillingMidCandidate._seen
                if seen > BillingMidCandidate.ok_candidate:
                    exc = type("BadRequestError", (Exception,), {"status_code": 400})(
                        "litellm.BadRequestError: Your credit balance is too low to "
                        "access the Anthropic API. Please go to Plans & Billing."
                    )
                    raise map_litellm_error(exc, request.model)
            return CompletionResponse(
                text='{"ok": 1}',
                tool_calls=None,
                finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1},
                latency_ms=1,
                served_model=request.model,
            )

    monkeypatch.setattr(cli, "LiveClient", BillingMidCandidate)
    monkeypatch.setattr(cli, "_missing_provider_keys", lambda _manifest: [])

    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    result = CliRunner().invoke(
        app,
        [*_base_args(manifest, run_dir, "billing"), "--live"],
    )

    assert result.exit_code == 1
    assert "credit balance is too low" in result.output
    active = run_dir / "runs" / "billing"
    assert not (active / "cert.json").exists()
    # Baseline checkpoints survive for resume after billing is fixed.
    assert (active / "baseline" / "responses.jsonl").exists()


def test_c_empty_candidate_stage_is_invalid_never_pass(tmp_path: Path) -> None:
    from tests.test_cert_build import _write_inputs

    manifest_path, manifest, traces, run_dir = _write_inputs(tmp_path)
    # Simulate the fail-open state: a stage that "completed" on rejected calls.
    (run_dir / "candidate" / "responses.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(EvidenceFloorError, match="run invalid"):
        build_certificate(
            manifest=manifest,
            traces=traces,
            run_dir=run_dir,
            manifest_path=manifest_path,
        )
    assert not (run_dir / "cert.json").exists()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
