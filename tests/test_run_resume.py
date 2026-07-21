from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

import acsi.cli as cli
from acsi.cli import app
from acsi.replay.clients import FakeClient, map_litellm_error

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _overloaded_error(model: str):
    exc = type("InternalServerError", (Exception,), {"status_code": 503})(
        "litellm.InternalServerError: Overloaded, please retry."
    )
    return map_litellm_error(exc, model)


class _CandidateFailsOnce(FakeClient):
    """Fake client that fails the first candidate call it sees, on every retry
    attempt (so retries exhaust into a permanent error), and serves all other
    calls normally. Baseline calls are untouched."""

    _CANDIDATE_MODEL = "claude-sonnet-5"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._target: str | None = None
        self._lock = threading.Lock()

    def complete(self, request):  # type: ignore[override]
        if request.model == self._CANDIDATE_MODEL:
            with self._lock:
                if self._target is None:
                    self._target = request.prompt_text
                fail = request.prompt_text == self._target
            if fail:
                raise _overloaded_error(request.model)
        return super().complete(request)


class _BaselineFailsOnce(FakeClient):
    """Fake client that fails the first baseline trace it sees (both k_baseline
    samples share the prompt), on every retry attempt; candidate calls untouched."""

    _BASELINE_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._target: str | None = None
        self._lock = threading.Lock()

    def complete(self, request):  # type: ignore[override]
        if request.model == self._BASELINE_MODEL:
            with self._lock:
                if self._target is None:
                    self._target = request.prompt_text
                fail = request.prompt_text == self._target
            if fail:
                raise _overloaded_error(request.model)
        return super().complete(request)


def _write_manifest(path: Path) -> Path:
    payload = {
        "assertions": [{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
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


def _run(args: list[str]):
    return CliRunner().invoke(app, args)


def test_run_resumes_most_recent_incomplete_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    common = [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]

    # Control run (isolated dir) provides the deterministic reference artifacts.
    control_dir = tmp_path / ".control"
    control = _run(
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(control_dir),
            "--fake-noise",
            "0.05",
            "--inject-broken-json-rate",
            "0.08",
            "--yes",
            "--json",
            "--run-id",
            "control",
        ]
    )
    assert control.exit_code == 0, control.output
    control_run = control_dir / "runs" / "control"

    # Interrupt mid-judging with no explicit run id.
    interrupted = _run([*common, "--interrupt-after-judge-dispatches", "2"])
    assert interrupted.exit_code == 1, interrupted.output
    runs = [p for p in (run_dir / "runs").iterdir() if p.is_dir()]
    assert len(runs) == 1
    first_run_id = runs[0].name
    assert not (runs[0] / "cert.json").exists()

    # Re-invoke with no flags: it resumes the incomplete run rather than starting new.
    resumed = _run(common)
    assert resumed.exit_code == 0, resumed.output
    payload = json.loads(resumed.output)
    assert payload["run_id"] == first_run_id
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 1

    # Deterministic artifacts match the uninterrupted control (run-id independent).
    resumed_run = run_dir / "runs" / first_run_id
    for artifact in ("judgments.jsonl", "assertion_results.jsonl", "clusters.json"):
        assert (resumed_run / artifact).read_bytes() == (control_run / artifact).read_bytes()


def test_fresh_forces_a_new_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    common = [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]

    interrupted = _run([*common, "--interrupt-after-judge-dispatches", "2"])
    assert interrupted.exit_code == 1
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 1

    # --fresh ignores the resumable run and starts a second one.
    fresh = _run([*common, "--fresh"])
    assert fresh.exit_code == 0, fresh.output
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 2


def _resumable_args(manifest: Path, run_dir: Path) -> list[str]:
    return [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--run-id",
        "gap",
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]


def test_resume_redispatches_permanently_failed_candidate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One candidate call fails permanently after retries, so the candidate stage
    # completes at n-1/n coverage and the evidence floor blocks with no cert.
    # Resuming with a healthy client must redispatch EXACTLY that one missing call
    # (banked responses reused) and issue the cert.
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    args = _resumable_args(manifest, run_dir)
    active = run_dir / "runs" / "gap"

    monkeypatch.setattr(cli, "FakeClient", _CandidateFailsOnce)
    blocked = _run(args)
    assert blocked.exit_code == 1, blocked.output
    assert "candidate responses cover" in blocked.output
    assert not (active / "cert.json").exists()
    n = _count_lines(active / "sampled_traces.jsonl")
    assert _count_lines(active / "candidate" / "responses.jsonl") == n - 1
    assert _stage(active, "replay")["status"] == "completed"  # completed-but-short

    monkeypatch.setattr(cli, "FakeClient", FakeClient)
    resumed = _run(args)
    assert resumed.exit_code == 0, resumed.output
    assert _stage(active, "replay")["dispatched"] == 1  # only the missing call
    assert (active / "cert.json").exists()
    assert _count_lines(active / "candidate" / "responses.jsonl") == n


def test_resume_redispatches_permanently_failed_baseline_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Baseline-stage variant: one trace's k_baseline samples fail permanently, so
    # the baseline stage completes without a pairable baseline response for that
    # trace and the floor blocks. Resume redispatches exactly those k_baseline
    # calls (candidate stage stays covered, dispatches nothing) and certifies.
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    args = _resumable_args(manifest, run_dir)
    active = run_dir / "runs" / "gap"

    monkeypatch.setattr(cli, "FakeClient", _BaselineFailsOnce)
    blocked = _run(args)
    assert blocked.exit_code == 1, blocked.output
    assert "run invalid" in blocked.output
    assert not (active / "cert.json").exists()

    monkeypatch.setattr(cli, "FakeClient", FakeClient)
    resumed = _run(args)
    assert resumed.exit_code == 0, resumed.output
    assert _stage(active, "baseline")["dispatched"] == 2  # the missing k_baseline pair
    assert (active / "cert.json").exists()


def _stage(active_run: Path, stage: str) -> dict:
    payload = json.loads((active_run / "run.json").read_text(encoding="utf-8"))
    return payload["stages"][stage]
