from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acsi.baseline import run_baseline
from acsi.cli import app
from acsi.importers.jsonl import import_jsonl_paths
from acsi.replay.clients import FakeClient
from acsi.replay.runner import ReplayAbortError, ReplayConfig
from acsi.replay.store import ReplayStore
from acsi.schemas import ProviderModel, TraceRecord

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
BASELINE_MODEL = ProviderModel(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
)


def test_baseline_noise_q005_ci_contains_analytic_note_and_is_stable(
    tmp_path: Path,
) -> None:
    first = _run_baseline(
        tmp_path / "first",
        noise=0.05,
        run_id="00000000-0000-0000-0000-000000000301",
    )
    second = _run_baseline(
        tmp_path / "second",
        noise=0.05,
        run_id="00000000-0000-0000-0000-000000000301",
    )

    ci = first.noise_floor["textual_mismatch_ci"]
    assert ci["lower"] <= 0.095 <= ci["upper"]
    assert first.noise_floor["analytic_note"] == {
        "expected_mismatch_rate": 0.095,
        "q": 0.05,
    }
    assert _noise_floor_bytes(first) == _noise_floor_bytes(second)


def test_baseline_zero_noise_uses_default_threshold(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        noise=0.0,
        run_id="00000000-0000-0000-0000-000000000302",
    )

    assert result.noise_floor["textual_mismatch_rate"] == 0.0
    assert result.noise_floor["tau"] == 0.9
    assert result.noise_floor["threshold_source"] == "default_insufficient_variation"


def test_baseline_calibrates_tau_and_beyond_noise_ratio(tmp_path: Path) -> None:
    result = _run_baseline(
        tmp_path,
        noise=0.3,
        run_id="00000000-0000-0000-0000-000000000303",
    )

    assert result.noise_floor["threshold_source"] == "calibrated"
    assert result.noise_floor["tau"] < 1.0
    ratio = result.noise_floor["beyond_noise_to_textual_mismatch_rate"]
    assert 0.02 <= ratio <= 0.10


def test_degraded_baseline_skips_retired_model_and_marks_run(tmp_path: Path) -> None:
    retired_model = ProviderModel(provider="anthropic", model="retired-model")
    retired_client = FakeClient(retired_models={"retired-model"})

    with pytest.raises(ReplayAbortError) as exc_info:
        _run_baseline(
            tmp_path / "aborted",
            model=retired_model,
            client=retired_client,
            run_id="00000000-0000-0000-0000-000000000304",
            trace_count=1,
        )
    assert str(exc_info.value) == (
        "Model retired-model returned 404; it may be retired. "
        "Rerun with --degraded to certify against stored outputs."
    )

    degraded = _run_baseline(
        tmp_path / "degraded",
        model=retired_model,
        client=retired_client,
        run_id="00000000-0000-0000-0000-000000000305",
        degraded=True,
        trace_count=3,
    )

    assert degraded.noise_floor["noise_floor"] == "unavailable"
    assert degraded.noise_floor["threshold_source"] == "default_degraded"
    run_payload = json.loads(
        (degraded.run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert run_payload["degraded"] is True


def test_baseline_cli_writes_noise_floor_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "acsi.yaml"
    traces = tmp_path / "traces.jsonl"
    _write_manifest(manifest, BASELINE_MODEL)
    _write_traces(traces, 3)

    result = CliRunner().invoke(
        app,
        [
            "baseline",
            "--manifest",
            str(manifest),
            "--traces",
            str(traces),
            "--run-id",
            "00000000-0000-0000-0000-000000000306",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    assert payload["threshold_source"] == "default_insufficient_variation"
    assert (run_dir / "baseline" / "noise_floor.json").exists()
    assert (run_dir / "responses.jsonl").exists()
    assert (run_dir / "run.json").exists()


def _run_baseline(
    root: Path,
    *,
    run_id: str,
    noise: float = 0.0,
    model: ProviderModel = BASELINE_MODEL,
    client: FakeClient | None = None,
    degraded: bool = False,
    trace_count: int = 300,
):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "acsi.yaml"
    _write_manifest(manifest, model)
    traces = _fixture_traces(trace_count)
    run_dir = root / "runs" / run_id
    active_client = client or FakeClient(seed=42, noise=noise)
    return asyncio.run(
        run_baseline(
            traces,
            model,
            2,
            client=active_client,
            store=ReplayStore(run_dir / "replay.sqlite"),
            config=ReplayConfig(run_id=run_id, seed=42, concurrency=4),
            run_dir=run_dir,
            manifest_path=manifest,
            traces_path=FIXTURE_PATH,
            endpoint="degraded" if degraded else "fake",
            degraded=degraded,
        )
    )


def _fixture_traces(count: int) -> list[TraceRecord]:
    return import_jsonl_paths([FIXTURE_PATH]).records[:count]


def _write_manifest(path: Path, model: ProviderModel) -> None:
    payload = {
        "assertions": [],
        "baseline": model.model_dump(mode="json"),
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {"families_allowed": ["openai"], "min_judges": 1},
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 300, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "volunteer-application-summary",
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _write_traces(path: Path, count: int) -> None:
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[:count]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _noise_floor_bytes(result) -> bytes:
    return (result.run_dir / "baseline" / "noise_floor.json").read_bytes()
