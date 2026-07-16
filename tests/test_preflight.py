from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acsi.cli import app
from acsi.preflight import run_preflight
from acsi.replay.clients import FakeClient
from acsi.schemas import WorkloadManifest

ALL_KEYS = {
    "ANTHROPIC_API_KEY": "test-anthropic",
    "OPENAI_API_KEY": "test-openai",
    "GEMINI_API_KEY": "test-google",
}
PROVIDER_KEY_ENVS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GITHUB_TOKEN")


def test_missing_keys_reported_by_name_without_any_completion() -> None:
    client = FakeClient()

    report = run_preflight(_manifest(), client=client, env={}, fake=True)

    assert report.ok is False
    assert report.missing_keys == ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]
    assert report.checks == []
    # No provider call may happen when keys are absent.
    assert client.call_count == 0


def test_all_ok_reports_every_distinct_model() -> None:
    report = run_preflight(_manifest(), client=FakeClient(), env=ALL_KEYS, fake=True)

    assert report.ok is True
    assert report.missing_keys == []
    served = {
        (check.provider, check.requested_model, check.served_model) for check in report.checks
    }
    assert served == {
        ("anthropic", "claude-opus-4-1", "claude-opus-4-1"),
        ("anthropic", "claude-sonnet-5", "claude-sonnet-5"),
        ("openai", "gpt-5.4-mini", "gpt-5.4-mini"),
        ("google", "gemini-3.5-flash", "gemini-3.5-flash"),
    }
    assert all(check.ok for check in report.checks)
    assert 0.0 <= report.estimated_cost_usd < 0.01


def test_retired_model_error_carries_404_hint() -> None:
    client = FakeClient(retired_models={"claude-sonnet-5"})

    report = run_preflight(_manifest(), client=client, env=ALL_KEYS, fake=True)

    assert report.ok is False
    failed = [check for check in report.checks if not check.ok]
    assert len(failed) == 1
    assert failed[0].requested_model == "claude-sonnet-5"
    assert "404" in failed[0].error
    assert "retired" in failed[0].error
    assert "--degraded" in failed[0].error


def test_json_payload_shape() -> None:
    payload = run_preflight(_manifest(), client=FakeClient(), env=ALL_KEYS, fake=True).to_payload()

    assert set(payload) == {
        "status",
        "ok",
        "missing_keys",
        "required_keys",
        "checks",
        "estimated_cost_usd",
    }
    assert payload["status"] == "ok"
    assert payload["required_keys"] == {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    first = payload["checks"][0]
    assert set(first) == {
        "role",
        "provider",
        "requested_model",
        "served_model",
        "latency_ms",
        "ok",
        "error",
    }


def test_cli_live_path_never_calls_network_without_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in PROVIDER_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)

    class _ExplodingClient:
        def complete(self, request: object) -> object:  # pragma: no cover - must never run
            raise AssertionError("preflight made a live provider call without keys present")

    monkeypatch.setattr("acsi.cli.LiveClient", _ExplodingClient)

    manifest_path = tmp_path / "acsi.json"
    manifest_path.write_text(
        json.dumps(_manifest().model_dump(mode="json")), encoding="utf-8"
    )

    result = CliRunner().invoke(
        app,
        ["preflight", "--manifest", str(manifest_path), "--json"],
    )

    assert result.exit_code == 1
    assert '"ok": false' in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def _manifest() -> WorkloadManifest:
    return WorkloadManifest.model_validate(
        {
            "assertions": [],
            "baseline": {"provider": "anthropic", "model": "claude-opus-4-1"},
            "budget": {"max_usd": 1.0, "use_batch_api": False},
            "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
            "judging": {
                "families_allowed": ["openai", "google", "local"],
                "judges": [
                    {"provider": "openai", "model": "gpt-5.4-mini"},
                    {"provider": "google", "model": "gemini-3.5-flash"},
                ],
                "min_judges": 2,
            },
            "privacy": {"egress": "hosted_api", "scrub": True},
            "sampling": {"k_baseline": 2, "n": 10, "seed": 42, "stratify_by": []},
            "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
            "workload": "demo",
        }
    )
