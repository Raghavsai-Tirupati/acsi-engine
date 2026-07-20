from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from acsi.cert.build import (
    BANNED_PHRASES,
    BannedLanguageError,
    Sanitizer,
    assert_no_banned_language,
)

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
REPORT_TEMPLATE = "report.html.j2"
ALPINE_PATH = TEMPLATE_DIR / "alpine.min.js"

_EXEMPLAR_PROMPT_CHARS = 600
_EXEMPLAR_RESPONSE_CHARS = 1400
_EVIDENCE_PER_CLUSTER = 3

# User-facing renames. The exact technical keys still appear in the auditor's
# raw view and the embedded certificate JSON, never in the plain narrative.
_SEVERITY_LABEL = {
    "worse_critical": "critical regression",
    "worse_minor": "minor regression",
    "major": "major",
    "minor": "minor",
    "unresolved": "unresolved",
}
_CRITERION_LABEL = {
    "critical_assertions": "Critical assertion failures",
    "candidate_regression_rate": "Regression vs. baseline noise",
    "critical_cluster_share": "Critical failure concentration",
}
_CRITERION_MEANING = {
    "critical_assertions": (
        "No sampled response may fail a critical check (schema, safety). "
        "Even one failure fails the run."
    ),
    "candidate_regression_rate": (
        "The candidate must not disagree with the baseline more than the baseline "
        "already disagrees with itself, plus a small tolerance."
    ),
    "critical_cluster_share": (
        "No single critical failure pattern may cover more than a small share of "
        "the sampled traffic."
    ),
}


def assert_no_banned_words(rendered: str) -> None:
    assert_no_banned_language(rendered)


def render_report(
    cert: dict[str, Any],
    *,
    output_path: Path,
    template_dir: Path = TEMPLATE_DIR,
    review_mode: bool = False,
    evidence_dir: Path | None = None,
) -> str:
    rendered = render_report_html(
        cert,
        template_dir=template_dir,
        review_mode=review_mode,
        evidence_dir=evidence_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        if not rendered.endswith("\n"):
            handle.write("\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    with Path(f"{output_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def render_report_html(
    cert: dict[str, Any],
    *,
    template_dir: Path = TEMPLATE_DIR,
    review_mode: bool = False,
    evidence_dir: Path | None = None,
) -> str:
    template_path = template_dir / REPORT_TEMPLATE
    alpine_path = template_dir / "alpine.min.js"
    template_source = template_path.read_text(encoding="utf-8")
    alpine_source = alpine_path.read_text(encoding="utf-8")
    assert_no_banned_language(template_source)
    assert_no_banned_language(alpine_source)

    env = Environment(
        autoescape=select_autoescape(("html", "xml")),
        loader=FileSystemLoader(template_dir),
    )
    env.filters["ci_pct"] = _format_ci_percent
    env.filters["decimal"] = _format_decimal
    env.filters["ms"] = _format_ms
    env.filters["multiplier"] = _format_multiplier
    env.filters["pct"] = _format_percent
    env.filters["pct_value"] = _format_percent_value
    env.filters["severity_label"] = _severity_label
    env.filters["usd"] = _format_usd
    template = env.get_template(REPORT_TEMPLATE)
    payload = cert["payload"]
    certificate_json = json.dumps(
        _html_json_value(cert),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    view = _build_view(payload)
    cluster_evidence = _load_cluster_evidence(evidence_dir, payload.get("clusters") or [])
    rendered = template.render(
        alpine_js=alpine_source,
        banned_phrases=", ".join(BANNED_PHRASES),
        cert=cert,
        certificate_json=certificate_json,
        cluster_evidence=cluster_evidence,
        payload=payload,
        review_mode=review_mode,
        view=view,
    )
    assert_no_banned_language(rendered)
    return rendered


def _build_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Precompute plain-English narratives and user-facing renames.

    Keeps rendering logic in Python (testable) and out of the template. Every
    number is formatted through the shared helpers so no raw long float ever
    reaches the page.
    """
    coverage = payload.get("coverage") or {}
    regressed = payload.get("regressed_pairs") or {}
    n = int(coverage.get("n") or 0)
    return {
        "is_pass": str(payload.get("verdict")) == "PASS",
        "regressed_headline": _regressed_headline(regressed, n),
        "unresolved_headline": _unresolved_headline(regressed, n),
        "criteria": [_criterion_card(criterion) for criterion in payload.get("criteria") or []],
        "noise_narrative": _noise_narrative(payload),
        "noise": _noise_values(payload),
        "judge_narrative": _judge_narrative(payload),
        "unresolved_reconciliation": _unresolved_reconciliation(payload),
    }


def _severity_label(severity: Any) -> str:
    return _SEVERITY_LABEL.get(str(severity), str(severity) or "—")


def _regressed_headline(regressed: dict[str, Any], n: int) -> str:
    count = int(regressed.get("count") or 0)
    by = regressed.get("by_source") or {}
    if count == 0:
        return f"No sampled pairs regressed across {n} inputs."
    sources = []
    if by.get("assertion"):
        sources.append(f"{int(by['assertion'])} by a failing assertion")
    if by.get("judge"):
        sources.append(f"{int(by['judge'])} by judge verdict")
    if by.get("both"):
        sources.append(f"{int(by['both'])} by both")
    tail = f" — {', '.join(sources)}." if sources else "."
    return (
        f"{count} of {n} sampled pairs ({_format_percent(regressed.get('rate', 0.0))}) "
        f"regressed{tail}"
    )


def _unresolved_headline(regressed: dict[str, Any], n: int) -> str | None:
    total = int(regressed.get("unresolved") or 0)
    if total == 0:
        return None
    only = int(regressed.get("unresolved_only") or 0)
    also = int(regressed.get("unresolved_also_regressed") or 0)
    return (
        f"{total} of {n} pairs ({_format_percent(regressed.get('unresolved_rate', 0.0))}) "
        f"were unresolved — the panel could not decide: {only} unresolved-only, "
        f"{also} also assertion-flagged. Counted conservatively toward the verdict, "
        "never as a judge-flagged regression."
    )


def _criterion_card(criterion: dict[str, Any]) -> dict[str, Any]:
    criterion_id = str(criterion.get("id"))
    passed = criterion.get("passed")
    return {
        "label": _CRITERION_LABEL.get(criterion_id, criterion_id),
        "meaning": _CRITERION_MEANING.get(criterion_id, ""),
        "input": _criterion_input(criterion),
        "threshold": _criterion_threshold(criterion),
        "status": "PASS" if passed is True else ("BLOCK" if passed is False else "n/a"),
        "is_pass": passed is True,
        "is_block": passed is False,
    }


def _criterion_input(criterion: dict[str, Any]) -> str:
    criterion_id = str(criterion.get("id"))
    if criterion_id == "critical_assertions":
        actual = int(criterion.get("actual", 0))
        return f"{actual} critical failure(s) observed"
    if criterion_id == "candidate_regression_rate":
        if criterion.get("mode") == "degraded":
            return "not evaluated — baseline noise floor unavailable"
        base = (
            f"candidate up to {_format_ci_percent(criterion.get('actual_ci_upper', 0.0))} "
            f"vs. allowed {_format_ci_percent(criterion.get('threshold', 0.0))} "
            f"(baseline noise {_format_ci_percent(criterion.get('baseline_ci_upper', 0.0))} "
            f"+ {_format_percent(criterion.get('epsilon', 0.0))} tolerance)"
        )
        unresolved = int(criterion.get("unresolved_pairs", 0) or 0)
        if unresolved:
            base += f"; includes {unresolved} unresolved pair(s), counted conservatively"
        return base
    if criterion_id == "critical_cluster_share":
        actual = criterion.get("actual") or []
        if not isinstance(actual, list) or not actual:
            return "no failure pattern above the concentration limit"
        parts = [
            f"one pattern at {_format_percent(item.get('share_of_sampled', 0.0))} of traffic"
            for item in actual
            if isinstance(item, dict)
        ]
        return "; ".join(parts)
    return ""


def _criterion_threshold(criterion: dict[str, Any]) -> str:
    criterion_id = str(criterion.get("id"))
    if criterion_id == "critical_assertions":
        return f"{int(criterion.get('threshold', 0))} allowed"
    if criterion.get("mode") == "degraded":
        return "not evaluated"
    if criterion_id in {"candidate_regression_rate", "critical_cluster_share"}:
        return f"{_format_percent(criterion.get('threshold', 0.0))} allowed"
    return str(criterion.get("threshold", "n/a"))


def _noise_narrative(payload: dict[str, Any]) -> str:
    noise = payload.get("noise_floor")
    if not noise:
        return (
            "The baseline noise floor was unavailable (degraded mode); the "
            "behavioral-variance comparison was not performed."
        )
    return (
        f"The baseline model disagrees with itself on "
        f"{_format_percent(noise.get('rate', 0.0))} of these inputs; only candidate "
        "differences beyond that bar count as real regressions."
    )


def _noise_values(payload: dict[str, Any]) -> dict[str, Any]:
    disagreement = payload.get("candidate_disagreement") or {}
    noise = payload.get("noise_floor") or {}
    raw = payload.get("noise_floor_raw") or {}
    return {
        "candidate_rate": _format_percent(disagreement.get("rate", 0.0)),
        "candidate_ci": (
            f"{_format_ci_percent(disagreement.get('lower', 0.0))} to "
            f"{_format_ci_percent(disagreement.get('upper', 0.0))}"
        ),
        "noise_rate": _format_percent(noise.get("rate", 0.0)) if noise else "unavailable",
        "similarity_threshold": _format_decimal(raw.get("tau", 0.9)),
        "bar_source": str(raw.get("threshold_source") or "unknown"),
    }


def _judge_narrative(payload: dict[str, Any]) -> str:
    panel = payload.get("judge_panel") or {}
    health = payload.get("judge_health") or {}
    models = panel.get("models") or []
    completed = int(panel.get("completed_pairs") or 0)
    if not models or completed == 0:
        return "No pairs needed judging; deterministic checks resolved every sampled pair."
    rate = health.get("valid_verdict_rate")
    rate_text = _format_percent(rate) if rate is not None else "n/a"
    return (
        f"{len(models)} judges graded the {completed} disputed pairs in both response "
        f"orders. Valid verdicts on {rate_text} of evaluations; the rest abstained on "
        "parse failures or order inconsistency."
    )


def _unresolved_reconciliation(payload: dict[str, Any]) -> str | None:
    regressed = payload.get("regressed_pairs") or {}
    only = int(regressed.get("unresolved_only") or 0)
    if only == 0:
        return None
    unresolved_clusters = [
        cluster
        for cluster in payload.get("clusters") or []
        if str(cluster.get("severity")) == "unresolved"
    ]
    if len(unresolved_clusters) <= 1:
        return None
    total = sum(int(cluster.get("count") or 0) for cluster in unresolved_clusters)
    if total != only:
        return None
    parts = " + ".join(str(int(cluster.get("count") or 0)) for cluster in unresolved_clusters)
    return (
        f"The {only} unresolved-only pairs split into {len(unresolved_clusters)} clusters "
        f"below ({parts} = {only})."
    )


def _load_cluster_evidence(
    evidence_dir: Path | None,
    clusters: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Per-cluster exemplar evidence (prompt, candidate response, reason) for the
    report's evidence layer.

    Loaded from the run's stored artifacts at render time — NOT from the signed
    certificate payload, which is unchanged. Returns {} when the artifacts are
    unavailable, so rendering degrades to the payload's prompt-only exemplars.
    """
    if evidence_dir is None:
        return {}
    stored = _read_json(evidence_dir / "clusters.json")
    if not stored:
        return {}
    prompts = _prompts_by_pair(evidence_dir / "sampled_traces.jsonl")
    responses = _responses_by_pair(evidence_dir / "candidate" / "responses.jsonl")
    reasons = _reasons_by_pair(evidence_dir / "assertion_results.jsonl")
    sanitizer = Sanitizer()
    by_cluster: dict[str, list[dict[str, str]]] = {}
    for cluster in stored.get("clusters", []):
        cluster_id = str(cluster.get("cluster_id"))
        rows: list[dict[str, str]] = []
        for pair_id in [str(pid) for pid in cluster.get("pair_ids", [])][:_EVIDENCE_PER_CLUSTER]:
            rows.append(
                {
                    "prompt": sanitizer.sanitize_text(
                        _truncate(prompts.get(pair_id, ""), _EXEMPLAR_PROMPT_CHARS)
                    ),
                    "response": sanitizer.sanitize_text(
                        _truncate(responses.get(pair_id, ""), _EXEMPLAR_RESPONSE_CHARS)
                    ),
                    "reason": sanitizer.sanitize_text(reasons.get(pair_id, "")),
                }
            )
        if rows:
            by_cluster[cluster_id] = rows
    return by_cluster


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _prompts_by_pair(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for record in _read_jsonl(path):
        messages = ((record.get("request") or {}).get("messages")) or []
        if messages:
            prompts[str(record.get("trace_id"))] = str(messages[0].get("content") or "")
    return prompts


def _responses_by_pair(path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    for record in _read_jsonl(path):
        text = (record.get("response") or {}).get("text")
        responses[str(record.get("trace_id"))] = str(text or "")
    return responses


def _reasons_by_pair(path: Path) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for record in _read_jsonl(path):
        reason = record.get("reason")
        pair_id = str(record.get("pair_id") or record.get("trace_id"))
        if reason and pair_id not in reasons:
            reasons[pair_id] = str(reason)
    return reasons


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _html_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_html_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _html_json_value(item) for key, item in value.items()}
    return value


def _format_percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def _format_percent_value(value: Any) -> str:
    return f"{float(value):.1f}%"


def _format_ci_percent(value: Any) -> str:
    percent = float(value) * 100
    digits = 2 if 0 < abs(percent) < 0.1 else 1
    return f"{percent:.{digits}f}%"


def _format_ms(value: Any) -> str:
    return f"{round(float(value))} ms"


def _format_usd(value: Any) -> str:
    return f"${float(value):.4f}"


def _format_multiplier(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}×"


def _format_decimal(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


__all__ = [
    "BannedLanguageError",
    "assert_no_banned_words",
    "render_report",
    "render_report_html",
]
