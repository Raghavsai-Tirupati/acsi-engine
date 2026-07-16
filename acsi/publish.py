from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class PublishError(ValueError):
    pass


@dataclass(frozen=True)
class PublishResult:
    status_code: int
    payload: dict[str, Any]
    response_text: str


def publish_certificate(
    cert_path: Path,
    *,
    url: str | None = None,
    include_examples: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> PublishResult:
    endpoint = url or os.environ.get("ACSI_PUBLISH_URL")
    if not endpoint:
        raise PublishError("Pass --url or set ACSI_PUBLISH_URL to publish a certificate.")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    payload = publish_payload(cert, include_examples=include_examples)
    with httpx.Client(transport=transport, timeout=30.0) as client:
        response = client.post(endpoint, json=payload)
    response.raise_for_status()
    return PublishResult(
        status_code=response.status_code,
        payload=payload,
        response_text=response.text,
    )


def publish_payload(cert: dict[str, Any], *, include_examples: bool) -> dict[str, Any]:
    payload = cert["payload"]
    clusters = []
    for cluster in payload.get("clusters", []):
        item = {
            "cluster_id": cluster.get("cluster_id"),
            "count": cluster.get("count"),
            "description": cluster.get("description"),
            "name": cluster.get("name"),
            "severity": cluster.get("severity"),
            "share_of_sampled": cluster.get("share_of_sampled"),
        }
        if include_examples:
            item["exemplars"] = cluster.get("exemplars", [])
            item["patch_diff"] = cluster.get("patch_diff")
        clusters.append(item)
    return {
        "clusters": clusters,
        "criteria": payload.get("criteria", []),
        "mode": payload.get("mode"),
        "run_id": payload.get("run_id"),
        "stats": {
            "candidate_disagreement": payload.get("candidate_disagreement"),
            "coverage": payload.get("coverage"),
            "cost_latency": payload.get("cost_latency"),
            "noise_floor": payload.get("noise_floor"),
        },
        "verdict": payload.get("verdict"),
    }
