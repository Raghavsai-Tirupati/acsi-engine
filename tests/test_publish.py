from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from acsi.publish import PublishError, publish_certificate, publish_payload


def test_publish_payload_strips_examples_by_default() -> None:
    payload = publish_payload(_cert(), include_examples=False)

    assert payload["verdict"] == "BLOCK"
    assert "exemplars" not in payload["clusters"][0]
    assert "patch_diff" not in payload["clusters"][0]


def test_publish_payload_can_include_cert_redacted_examples_only() -> None:
    payload = publish_payload(_cert(), include_examples=True)

    assert payload["clusters"][0]["exemplars"] == ["[EMAIL_1]"]
    assert payload["clusters"][0]["patch_diff"] == "--- patch"


def test_publish_certificate_uses_mock_transport(tmp_path: Path) -> None:
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(_cert(), sort_keys=True), encoding="utf-8")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(202, text="accepted")

    result = publish_certificate(
        cert_path,
        url="https://publish.test/cert",
        transport=httpx.MockTransport(handler),
    )

    assert result.status_code == 202
    assert seen[0]["verdict"] == "BLOCK"


def test_publish_certificate_requires_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACSI_PUBLISH_URL", raising=False)
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(_cert(), sort_keys=True), encoding="utf-8")

    with pytest.raises(PublishError, match="Pass --url"):
        publish_certificate(cert_path)


def _cert() -> dict:
    return {
        "payload": {
            "candidate_disagreement": {"rate": 0.08},
            "clusters": [
                {
                    "cluster_id": "cluster-0",
                    "count": 8,
                    "description": "Broken JSON",
                    "exemplars": ["[EMAIL_1]"],
                    "name": "Broken JSON",
                    "patch_diff": "--- patch",
                    "severity": "worse_critical",
                    "share_of_sampled": 0.08,
                }
            ],
            "cost_latency": {"tokenizer_inflation": 1.2},
            "coverage": {"n": 100},
            "criteria": [{"id": "critical_assertions", "passed": False}],
            "mode": "standard",
            "noise_floor": {"upper": 0.01},
            "run_id": "run-1",
            "verdict": "BLOCK",
        }
    }
