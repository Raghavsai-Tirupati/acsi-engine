from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from acsi.importers.jsonl import import_jsonl_paths
from acsi.monitor import init_monitor, run_monitor
from acsi.replay.clients import CompletionRequest, FakeClient, PermanentError, RegressionRule
from acsi.replay.runner import ReplayInterrupted, build_completion_request
from acsi.schemas import WorkloadManifest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
RUN_ID = "00000000-0000-0000-0000-000000000702"


def test_monitor_init_writes_deterministic_golden_suite(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])

    first = init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    second = init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    manifest_payload = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    assert first.suite_size == 6
    assert first.suite_hash == second.suite_hash
    assert first.golden_path.exists()
    assert manifest_payload["pinned_model"] == {"provider": "anthropic", "model": "claude-old"}
    assert manifest_payload["stored_noise_floor_ci"]["upper"] == 0.0
    assert manifest_payload["tau"] == 0.9


def test_monitor_init_includes_all_assertion_bearing_prompts(tmp_path: Path) -> None:
    manifest, run_root, active_run_dir = _write_certified_run(
        tmp_path,
        assertions=[{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        assertion_trace_indexes=[0, 2, 4],
    )

    result = init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)

    assertion_bearing = {
        json.loads(line)["pair_id"]
        for line in (active_run_dir / "assertion_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    }
    included = {
        json.loads(line)["trace_id"]
        for line in result.golden_path.read_text(encoding="utf-8").splitlines()
    }
    assert len(assertion_bearing) == 3
    assert len(assertion_bearing & included) == 3
    assert assertion_bearing <= included


def test_monitor_run_clean_returns_no_drift_summary(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)

    result = run_monitor(manifest=manifest, run_dir=run_root, run_id="monitor-clean")

    assert result.exit_code == 0
    assert result.summary == {
        "assertion_regressions": [],
        "beyond_noise_rate": 0.0,
        "drift": False,
        "pinned_model": "claude-old",
        "reasons": [],
        "served_model": "claude-old",
        "stored_bound": 0.0,
    }
    assert result.summary_path is not None and result.summary_path.exists()
    assert result.summary_sha256


def test_monitor_drift_on_beyond_noise_rate(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    client = FakeClient(
        seed=42,
        regressions=[
            RegressionRule(
                predicate=lambda _prompt: True,
                transform=lambda _prompt, _text: "unrelated zebra quantum failure mode",
            )
        ],
    )

    result = run_monitor(
        manifest=manifest,
        run_dir=run_root,
        client=client,
        run_id="monitor-noise-drift",
    )

    assert result.exit_code == 2
    assert result.summary["drift"] is True
    assert result.summary["beyond_noise_rate"] > result.summary["stored_bound"]
    assert result.summary["reasons"] == ["beyond_noise_rate"]


def test_monitor_drift_on_critical_assertion_regression(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(
        tmp_path,
        assertions=[{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
    )
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    client = FakeClient(
        seed=42,
        regressions=[
            RegressionRule(
                predicate=lambda _prompt: True,
                transform=lambda _prompt, _text: "{broken json",
            )
        ],
    )

    result = run_monitor(
        manifest=manifest,
        run_dir=run_root,
        client=client,
        run_id="monitor-assertion-drift",
    )

    assert result.exit_code == 2
    assert "critical_assertion_regression" in result.summary["reasons"]
    assert result.summary["assertion_regressions"] == ["json-valid"]


def test_monitor_drift_on_served_model_mismatch(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    client = FakeClient(seed=42, served_model_override="claude-newer-alias")

    result = run_monitor(
        manifest=manifest,
        run_dir=run_root,
        client=client,
        run_id="monitor-model-drift",
    )

    assert result.exit_code == 2
    assert result.summary["reasons"] == ["served_model_mismatch"]
    assert result.summary["served_model"] == "claude-newer-alias"
    assert result.summary["pinned_model"] == "claude-old"


def test_monitor_retired_pinned_model_is_drift_exit_two(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)
    client = FakeClient(seed=42, retired_models={"claude-old"})

    result = run_monitor(
        manifest=manifest,
        run_dir=run_root,
        client=client,
        run_id="monitor-retired-pinned-model",
    )

    assert result.exit_code == 2
    assert result.operational_message is None
    assert result.summary["reasons"] == ["model_retired"]
    assert result.summary["message"] == (
        "Pinned model claude-old returned 404; it may be retired or renamed."
    )
    assert result.summary_path is not None and result.summary_path.exists()


def test_monitor_unauthorized_run_level_failure_stays_operational(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)

    result = run_monitor(
        manifest=manifest,
        run_dir=run_root,
        client=UnauthorizedClient(),
        run_id="monitor-unauthorized",
    )

    assert result.exit_code == 1
    assert result.operational_message == "monitor operational failure: Unauthorized monitor replay."
    assert result.summary_path is None


def test_monitor_checkpoint_resume_reuses_completed_calls(tmp_path: Path) -> None:
    manifest, run_root, _active_run_dir = _write_certified_run(tmp_path, assertions=[])
    init_monitor(manifest=manifest, run_id=RUN_ID, run_dir=run_root)

    with pytest.raises(ReplayInterrupted):
        run_monitor(
            manifest=manifest,
            run_dir=run_root,
            run_id="monitor-resume",
            interrupt_after_dispatches=3,
        )
    resumed = run_monitor(manifest=manifest, run_dir=run_root, run_id="monitor-resume")
    control = run_monitor(manifest=manifest, run_dir=run_root, run_id="monitor-control")

    assert resumed.cache_hits >= 3
    assert resumed.dispatched + resumed.cache_hits == 6
    assert resumed.summary == control.summary


def _write_certified_run(
    tmp_path: Path,
    *,
    assertions: list[dict[str, Any]],
    assertion_trace_indexes: list[int] | None = None,
) -> tuple[WorkloadManifest, Path, Path]:
    manifest_payload = {
        "assertions": assertions,
        "baseline": {"provider": "anthropic", "model": "claude-old"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-new"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/judge-a"}],
            "min_judges": 1,
        },
        "monitor": {"suite_size": 6},
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 6, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "demo",
    }
    manifest = WorkloadManifest.model_validate(manifest_payload)
    traces = import_jsonl_paths([FIXTURE_PATH]).records[:6]
    run_root = tmp_path / ".acsi"
    active_run_dir = run_root / "runs" / RUN_ID
    (active_run_dir / "baseline").mkdir(parents=True)
    _write_jsonl(
        active_run_dir / "sampled_traces.jsonl",
        [trace.model_dump(mode="json") for trace in traces],
    )
    _write_jsonl(
        active_run_dir / "baseline" / "responses.jsonl",
        [_baseline_response(trace, manifest.baseline) for trace in traces],
    )
    assertion_indexes = assertion_trace_indexes or ([0] if assertions else [])
    _write_jsonl(
        active_run_dir / "assertion_results.jsonl",
        [
            {
                "assertion_id": assertion["id"],
                "baseline_passed": True,
                "candidate_passed": True,
                "pair_id": str(traces[index].trace_id),
                "severity": assertion["severity"],
            }
            for assertion in assertions
            for index in assertion_indexes
        ],
    )
    _write_json(
        active_run_dir / "cert.json",
        {
            "payload": {
                "noise_floor_raw": {
                    "beyond_noise_ci": {
                        "confidence": 0.95,
                        "lower": 0.0,
                        "rate": 0.0,
                        "upper": 0.0,
                    },
                    "tau": 0.9,
                    "threshold_source": "test",
                }
            }
        },
    )
    return manifest, run_root, active_run_dir


class UnauthorizedClient:
    def complete(self, _request: CompletionRequest):
        raise PermanentError(
            "Unauthorized monitor replay.",
            run_level=True,
            status_code=401,
        )


def _baseline_response(trace, model) -> dict[str, Any]:
    request, _prompt_hash, _params_hash, _transforms = build_completion_request(trace, model, 0)
    response = FakeClient(seed=42).complete(request)
    return {
        "cost_usd": 0.0,
        "model": model.model,
        "response": {
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
            "served_model": response.served_model,
            "text": response.text,
            "tool_calls": response.tool_calls,
        },
        "retry_count": 0,
        "sample_index": 0,
        "served_model": response.served_model,
        "status": "done",
        "trace_id": str(trace.trace_id),
        "usage": response.usage,
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
