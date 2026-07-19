from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from acsi import __version__
from acsi.overrides import (
    aggregate_judgment_rows,
    apply_overrides_to_judgments,
    human_overrides_payload,
    latest_overrides,
    read_overrides,
)
from acsi.replay.artifacts import sha256_file
from acsi.schemas import TraceRecord, WorkloadManifest
from acsi.stats import percentile_bootstrap_ci, rule_of_three_upper_bound

BANNED_PHRASES = ("guarantee", "guaranteed", "identical", "zero risk", "proven equivalent")
BANNED_RE = re.compile(
    r"\b(?:guarantee|guaranteed|identical|zero\ risk|proven\ equivalent)\b",
    re.IGNORECASE,
)
REGRESSION_OUTCOMES = {"worse_minor", "worse_critical", "unresolved"}
SCHEMA_VERSION = "1.0"


class BannedLanguageError(ValueError):
    pass


class CertificateVerificationError(ValueError):
    pass


class EvidenceFloorError(ValueError):
    # SPEC-NOTE: this certifier fails closed. Absence of evidence is a run
    # failure, never a pass: a verdict may only be issued when the sampled pairs
    # are actually covered by candidate responses (and, transitively, assertion
    # evaluations). A stage that "completed" on rejected provider calls yields no
    # certificate, never PASS and never BLOCK.
    pass


@dataclass(frozen=True)
class BuildCertificateResult:
    cert: dict[str, Any]
    cert_sha256: str
    payload: dict[str, Any]
    key_generated: bool = False


def build_certificate(
    *,
    manifest: WorkloadManifest,
    traces: list[TraceRecord],
    run_dir: Path,
    manifest_path: Path,
    cert_path: Path | None = None,
    degraded: bool = False,
    client_mode: str = "fake",
    authored_context: list[str] | None = None,
) -> BuildCertificateResult:
    assert_authored_strings_clean(*(authored_context or []))
    if not traces:
        raise ValueError("Cannot build a certificate for zero sampled traces.")

    active_cert_path = cert_path or run_dir / "cert.json"
    run_payload = _read_json(run_dir / "run.json", default={})
    run_started_at = str(run_payload.get("run_started_at") or _stable_now())
    run_id = str(run_payload.get("run_id") or run_dir.name)
    sampling_report = _read_json(run_dir / "sampling_report.json", default={})
    scrub_report = _read_json(run_dir / "scrub_report.json", default={})
    noise_floor = _read_json(run_dir / "baseline" / "noise_floor.json", default={})
    assertion_rows = _read_jsonl(run_dir / "assertion_results.jsonl")
    judgment_rows = _read_jsonl(run_dir / "judgments.jsonl")
    override_rows = read_overrides(run_dir)
    effective_judgment_rows = apply_overrides_to_judgments(judgment_rows, override_rows)
    judge_stats = _read_json(run_dir / "judge_stats.json", default={})
    clusters_payload = _read_json(run_dir / "clusters.json", default={"clusters": []})
    patches_payload = _read_json(
        run_dir / "patches" / "patch_report.json",
        default={"patches": []},
    )
    baseline_calls = _read_jsonl(_first_existing([run_dir / "baseline" / "responses.jsonl"]))
    candidate_calls = _read_jsonl(_first_existing([run_dir / "candidate" / "responses.jsonl"]))

    # Fail closed: refuse to issue any verdict unless every sampled pair has a
    # candidate response. The single chokepoint that no verdict path can bypass.
    _assert_candidate_coverage(traces, candidate_calls)

    n = len(traces)
    # SPEC-NOTE: reclassify pairs below the panel floor to "unresolved" so a lone
    # surviving judge cannot decide a pair when min_judges was required (run #1's
    # judge-error fragility). Human overrides are authoritative and bypass the
    # floor — they are overlaid after aggregation.
    judge_outcomes = aggregate_judgment_rows(
        effective_judgment_rows,
        min_valid=manifest.judging.min_judges,
    )
    for pair_id, override in latest_overrides(override_rows).items():
        judge_outcomes[str(pair_id)] = str(override["to_outcome"])
    unresolved_pairs = sum(1 for outcome in judge_outcomes.values() if outcome == "unresolved")
    candidate_ci = _candidate_regression_ci(
        judge_outcomes,
        traces,
        manifest.sampling.seed,
    )
    regressed_pairs = _regressed_pairs(assertion_rows, judge_outcomes, n=len(traces))
    noise_ci = _noise_floor_ci(noise_floor)
    degraded_mode = degraded or bool(noise_floor.get("degraded")) or noise_ci is None
    critical_failures = _critical_failure_count(assertion_rows)
    criterion_a = {
        "actual": critical_failures,
        "id": "critical_assertions",
        "passed": critical_failures <= manifest.thresholds.max_critical,
        "threshold": manifest.thresholds.max_critical,
    }
    if degraded_mode:
        criterion_b = {
            "id": "candidate_regression_rate",
            "mode": "degraded",
            "passed": None,
            "reason": "noise_floor_unavailable",
        }
    else:
        assert noise_ci is not None
        threshold = noise_ci["upper"] + manifest.thresholds.epsilon_pp / 100
        criterion_b = {
            "actual_ci_upper": candidate_ci["upper"],
            "baseline_ci_upper": noise_ci["upper"],
            "epsilon": manifest.thresholds.epsilon_pp / 100,
            "id": "candidate_regression_rate",
            "unresolved_pairs": unresolved_pairs,
            "passed": candidate_ci["upper"] <= threshold,
            "threshold": threshold,
        }
    critical_clusters = _critical_clusters(
        clusters_payload,
        assertion_rows,
        judge_outcomes,
        n=n,
    )
    criterion_c = {
        "actual": [
            {
                "cluster_id": cluster.get("cluster_id"),
                "share_of_sampled": cluster.get("share_of_sampled"),
            }
            for cluster in critical_clusters
        ],
        "id": "critical_cluster_share",
        "passed": not critical_clusters,
        "threshold": 0.01,
    }
    verdict = _verdict([criterion_a, criterion_b, criterion_c], degraded_mode=degraded_mode)

    sanitizer = Sanitizer()
    clusters = _certificate_clusters(
        clusters_payload,
        patches_payload,
        traces,
        assertion_reasons=_assertion_reasons_by_pair(assertion_rows),
        sanitizer=sanitizer,
    )
    coverage_sentence = _coverage_sentence(
        verdict=verdict,
        n=n,
        pct=_coverage_percent(sampling_report, traces),
        ci=candidate_ci,
        degraded=degraded_mode,
    )
    zero_event_sentence = _zero_event_sentence(n) if critical_failures == 0 else None
    human_overrides = human_overrides_payload(override_rows)
    coverage: dict[str, Any] = {
        "exclusion_percent": _exclusion_percent(sampling_report),
        "n": n,
        "production_template_coverage_pct": _coverage_percent(sampling_report, traces),
        "sampling_method": str(sampling_report.get("sampling_mode") or "unknown"),
        "strata": sampling_report.get("strata", []),
        "zero_event_bound_sentence": zero_event_sentence,
        **_dedup_scope(sampling_report),
    }
    if human_overrides["count"]:
        count = human_overrides["count"]
        coverage["human_override_footnote"] = (
            f"{count} judge outcome(s) were overridden by human review; "
            "original judge output is preserved in the run record."
        )
    payload: dict[str, Any] = {
        "accepted_patches": [
            cluster["patch_diff"]
            for cluster in clusters
            if cluster.get("patch_diff")
        ],
        "assertions_by_severity": _assertions_by_severity(assertion_rows, manifest),
        # SPEC-NOTE: the live-gap task asked for `mode: "fake"` in the payload, but
        # the top-level `mode` key already carries the degraded/standard verdict
        # context (asserted in tests, read by publish.py). To keep verdict
        # machinery untouched, the fake/live client watermark lives in a dedicated
        # `client_mode` key ("fake" or "live") that the report banner keys off.
        "client_mode": client_mode,
        "candidate_disagreement": candidate_ci,
        "candidate_regression_rate": candidate_ci["rate"],
        "clusters": clusters,
        "config_hash": sha256_file(manifest_path),
        "cost_latency": _cost_latency_payload(baseline_calls, candidate_calls),
        "coverage": coverage,
        "coverage_sentence": coverage_sentence,
        "criteria": [criterion_a, criterion_b, criterion_c],
        "delta": _delta_ci(candidate_ci, noise_ci),
        "engine_version": __version__,
        "human_overrides": human_overrides,
        "judge_health": _judge_health(judge_stats),
        "judge_panel": _judge_panel(judge_stats, judged=bool(judgment_rows)),
        "manifest": {
            "baseline": manifest.baseline.model_dump(mode="json"),
            "candidate": manifest.candidate.model_dump(mode="json"),
            "workload": manifest.workload,
        },
        "mode": "degraded" if degraded_mode else "standard",
        "noise_floor": noise_ci,
        "noise_floor_raw": noise_floor,
        "regressed_pairs": regressed_pairs,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "scope": {
            "sampled_trace_hash": run_payload.get("sampled_trace_hash"),
            "scrubbed": bool(manifest.privacy.scrub),
            "scrub_report": scrub_report,
            "workload": manifest.workload,
        },
        "verdict": verdict,
    }
    payload = sanitizer.sanitize_payload(payload)
    payload["banned_language_sanitization_count"] = sanitizer.count
    assert_no_banned_language(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    private_key, public_key, generated = _load_or_create_signing_key(run_dir)
    signature = _sign_payload(private_key, payload)
    cert = {
        "header": {
            "algo": "ed25519",
            "engine_version": __version__,
            "issued_at": _stable_now(),
            "public_key": _public_key_b64(public_key),
            "schema_version": SCHEMA_VERSION,
        },
        "payload": payload,
        "signature": signature,
    }
    assert_no_banned_language(json.dumps(cert, sort_keys=True, ensure_ascii=False))
    digest = _write_json(active_cert_path, cert)
    return BuildCertificateResult(
        cert=cert,
        cert_sha256=digest,
        payload=payload,
        key_generated=generated,
    )


class Sanitizer:
    def __init__(self) -> None:
        self.count = 0

    def sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, list):
            return [self.sanitize_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self.sanitize_payload(item) for key, item in value.items()}
        return value

    def sanitize_text(self, value: str) -> str:
        matches = BANNED_RE.findall(value)
        self.count += len(matches)
        return BANNED_RE.sub("[term removed]", value)


def assert_authored_strings_clean(*values: str) -> None:
    for value in values:
        assert_no_banned_language(value)


def assert_no_banned_language(value: str) -> None:
    match = BANNED_RE.search(value)
    if match:
        raise BannedLanguageError(f"Certificate contains banned wording: {match.group(0)}")


def verify_certificate(cert_path: Path) -> dict[str, Any]:
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(cert["header"]["public_key"])
        )
        signature = base64.b64decode(cert["signature"])
        public_key.verify(signature, canonical_payload_bytes(cert["payload"]))
    except (KeyError, InvalidSignature, ValueError) as exc:
        raise CertificateVerificationError("Certificate signature verification failed.") from exc
    return cert


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sign_payload(private_key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return base64.b64encode(private_key.sign(canonical_payload_bytes(payload))).decode("ascii")


def _load_or_create_signing_key(run_dir: Path) -> tuple[Ed25519PrivateKey, Ed25519PublicKey, bool]:
    env_value = os.environ.get("ACSI_SIGNING_KEY")
    if env_value:
        private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(env_value))
        return private_key, private_key.public_key(), False

    key_path = run_dir.parents[1] / "keys" / "ed25519.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        raw = base64.b64decode(key_path.read_text(encoding="utf-8").strip())
        private_key = Ed25519PrivateKey.from_private_bytes(raw)
        return private_key, private_key.public_key(), False

    private_key = Ed25519PrivateKey.generate()
    raw = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    with key_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{base64.b64encode(raw).decode('ascii')}\n")
    with suppress(OSError):
        key_path.chmod(0o600)
    return private_key, private_key.public_key(), True


def _public_key_b64(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def _candidate_regression_ci(
    judge_outcomes: dict[str, Any],
    traces: list[TraceRecord],
    seed: int,
) -> dict[str, float]:
    indicators = [
        float(judge_outcomes.get(str(trace.trace_id), "equivalent") in REGRESSION_OUTCOMES)
        for trace in traces
    ]
    ci = percentile_bootstrap_ci(indicators or [0.0], seed=seed)
    return _ci_payload(ci.mean, ci.lower, ci.upper, ci.confidence)


def _regressed_pairs(
    assertion_rows: list[dict[str, Any]],
    judge_outcomes: dict[str, Any],
    *,
    n: int,
) -> dict[str, Any]:
    # SPEC-NOTE: "regressed" counts only genuine evidence of harm — a failing
    # assertion or a judge verdict of worse_minor/worse_critical. Pairs the panel
    # could not decide are counted separately as "unresolved" (still conservative
    # for the verdict via the candidate CI) and are never laundered into the
    # judge-flagged regression count, which run 0a716021 did to reach 85.3%.
    assertion_pairs = {
        str(row.get("pair_id") or row.get("trace_id"))
        for row in assertion_rows
        if str(row.get("severity")) in {"critical", "major"}
        and row.get("baseline_passed") is True
        and row.get("candidate_passed") is False
    }
    judge_worse = {
        pair_id
        for pair_id, outcome in judge_outcomes.items()
        if outcome in {"worse_minor", "worse_critical"}
    }
    unresolved = {
        pair_id
        for pair_id, outcome in judge_outcomes.items()
        if outcome == "unresolved"
    }
    both = assertion_pairs & judge_worse
    assertion_only = assertion_pairs - judge_worse
    judge_only = judge_worse - assertion_pairs
    count = len(assertion_only) + len(judge_only) + len(both)
    # SPEC-NOTE: run 7f0978f5 showed two different numbers for one concept — the
    # headline said "66 unresolved" (panel-unresolved minus assertion overlap)
    # while criterion B said "128 unresolved" (all panel-unresolved). One taxonomy
    # now: `unresolved` is the total the panel could not decide; `unresolved_only`
    # excludes pairs already counted as regressions; `unresolved_also_regressed` is
    # the overlap. count + unresolved_only + (clean) sum to n, and the overlap is
    # visible rather than silently dropped.
    unresolved_also_regressed = unresolved & (assertion_pairs | judge_worse)
    unresolved_only = unresolved - assertion_pairs - judge_worse
    return {
        "by_source": {
            "assertion": len(assertion_only),
            "both": len(both),
            "judge": len(judge_only),
        },
        "count": count,
        "rate": round(count / n, 12) if n else 0.0,
        "unresolved": len(unresolved),
        "unresolved_also_regressed": len(unresolved_also_regressed),
        "unresolved_only": len(unresolved_only),
        "unresolved_rate": round(len(unresolved) / n, 12) if n else 0.0,
    }


def _critical_clusters(
    clusters_payload: dict[str, Any],
    assertion_rows: list[dict[str, Any]],
    judge_outcomes: dict[str, Any],
    *,
    n: int,
) -> list[dict[str, Any]]:
    assertion_pairs = {
        str(row.get("pair_id") or row.get("trace_id"))
        for row in assertion_rows
        if str(row.get("severity")) == "critical"
        and row.get("baseline_passed") is True
        and row.get("candidate_passed") is False
    }
    active_critical_pairs = {
        pair_id
        for pair_id, outcome in judge_outcomes.items()
        if outcome == "worse_critical"
    } | assertion_pairs
    clusters: list[dict[str, Any]] = []
    for cluster in clusters_payload.get("clusters", []):
        if cluster.get("severity") != "worse_critical":
            continue
        pair_ids = {str(pair_id) for pair_id in cluster.get("pair_ids", [])}
        if not pair_ids:
            if float(cluster.get("share_of_sampled", 0.0)) > 0.01:
                clusters.append(dict(cluster))
            continue
        legacy_cluster_only_pairs = pair_ids - set(judge_outcomes) - assertion_pairs
        active_pairs = (pair_ids & active_critical_pairs) | legacy_cluster_only_pairs
        share = len(active_pairs) / n if n else 0.0
        if share > 0.01:
            copied = dict(cluster)
            copied["share_of_sampled"] = round(share, 12)
            copied["member_count"] = len(active_pairs)
            copied["pair_ids"] = sorted(active_pairs)
            clusters.append(copied)
    return clusters


def _noise_floor_ci(noise_floor: dict[str, Any]) -> dict[str, float] | None:
    if noise_floor.get("degraded") or noise_floor.get("noise_floor") == "unavailable":
        return None
    raw = noise_floor.get("beyond_noise_ci")
    if not isinstance(raw, dict):
        return None
    return {
        "confidence": float(raw.get("confidence", 0.95)),
        "lower": float(raw.get("lower", 0.0)),
        "rate": float(raw.get("rate", raw.get("mean", 0.0))),
        "upper": float(raw.get("upper", 0.0)),
    }


def _ci_payload(rate: float, lower: float, upper: float, confidence: float) -> dict[str, float]:
    return {
        "confidence": round(float(confidence), 12),
        "lower": round(float(lower), 12),
        "rate": round(float(rate), 12),
        "upper": round(float(upper), 12),
    }


def _delta_ci(candidate_ci: dict[str, float], noise_ci: dict[str, float] | None) -> dict[str, Any]:
    if noise_ci is None:
        return {"mode": "degraded", "rate": None}
    return {
        "confidence": candidate_ci["confidence"],
        "lower": round(candidate_ci["lower"] - noise_ci["upper"], 12),
        "rate": round(candidate_ci["rate"] - noise_ci["rate"], 12),
        "upper": round(candidate_ci["upper"] - noise_ci["lower"], 12),
    }


def _verdict(criteria: list[dict[str, Any]], *, degraded_mode: bool) -> str:
    for criterion in criteria:
        if criterion.get("passed") is False:
            return "BLOCK"
        if criterion.get("passed") is None and not degraded_mode:
            return "BLOCK"
    return "PASS"


def _critical_failure_count(assertion_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in assertion_rows
        if row.get("severity") == "critical" and row.get("candidate_passed") is False
    )


def _assertions_by_severity(
    assertion_rows: list[dict[str, Any]],
    manifest: WorkloadManifest,
) -> dict[str, Any]:
    configured = {
        severity: [
            assertion.id for assertion in manifest.assertions if assertion.severity == severity
        ]
        for severity in ("critical", "major", "minor")
    }
    failures: dict[str, int] = {"critical": 0, "major": 0, "minor": 0}
    for row in assertion_rows:
        severity = str(row.get("severity"))
        if severity in failures and row.get("candidate_passed") is False:
            failures[severity] += 1
    return {
        severity: {
            "configured": configured[severity],
            "failures": failures[severity],
        }
        for severity in ("critical", "major", "minor")
    }


def _assert_candidate_coverage(
    traces: list[TraceRecord],
    candidate_calls: list[dict[str, Any]],
) -> None:
    sampled_ids = {str(trace.trace_id) for trace in traces}
    covered = {
        str(call.get("trace_id"))
        for call in candidate_calls
        if int(call.get("sample_index", 0)) == 0
    }
    missing = sampled_ids - covered
    if missing:
        raise EvidenceFloorError(
            "run invalid: candidate responses cover "
            f"{len(sampled_ids) - len(missing)}/{len(sampled_ids)} sampled pairs; "
            "refusing to issue a verdict"
        )


def _certificate_clusters(
    clusters_payload: dict[str, Any],
    patches_payload: dict[str, Any],
    traces: list[TraceRecord],
    *,
    assertion_reasons: dict[str, list[str]],
    sanitizer: Sanitizer,
) -> list[dict[str, Any]]:
    traces_by_id = {str(trace.trace_id): trace for trace in traces}
    accepted_patches = _accepted_patches(patches_payload)
    clusters: list[dict[str, Any]] = []
    for cluster in sorted(
        clusters_payload.get("clusters", []),
        key=lambda item: str(item.get("cluster_id", "")),
    ):
        pair_ids = [str(pair_id) for pair_id in cluster.get("pair_ids", [])]
        exemplars = [
            _truncate_exemplar(traces_by_id[pair_id].request.messages[0].content)
            for pair_id in pair_ids
            if pair_id in traces_by_id
        ][:3]
        cluster_reasons = _distinct_cluster_reasons(pair_ids, assertion_reasons)
        cluster_id = str(cluster.get("cluster_id"))
        patch_diff = accepted_patches.get(cluster_id)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "count": int(cluster.get("member_count", len(pair_ids))),
                "description": sanitizer.sanitize_text(str(cluster.get("description", ""))),
                "exemplars": [sanitizer.sanitize_text(exemplar) for exemplar in exemplars],
                "name": sanitizer.sanitize_text(str(cluster.get("name", ""))),
                "patch_diff": sanitizer.sanitize_text(patch_diff) if patch_diff else None,
                "reasons": [sanitizer.sanitize_text(reason) for reason in cluster_reasons],
                "severity": cluster.get("severity"),
                "share_of_sampled": cluster.get("share_of_sampled"),
            }
        )
    return clusters


def _assertion_reasons_by_pair(assertion_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for row in assertion_rows:
        reason = row.get("reason")
        if not reason:
            continue
        pair_id = str(row.get("pair_id") or row.get("trace_id"))
        reasons.setdefault(pair_id, []).append(str(reason))
    return reasons


def _distinct_cluster_reasons(
    pair_ids: list[str],
    assertion_reasons: dict[str, list[str]],
    *,
    limit: int = 5,
) -> list[str]:
    seen: list[str] = []
    for pair_id in pair_ids:
        for reason in assertion_reasons.get(pair_id, []):
            if reason not in seen:
                seen.append(reason)
            if len(seen) >= limit:
                return seen
    return seen


def _accepted_patches(patches_payload: dict[str, Any]) -> dict[str, str]:
    accepted: dict[str, str] = {}
    for patch in patches_payload.get("patches", []):
        if not patch.get("accepted"):
            continue
        diff_path = Path(str(patch["diff_path"]))
        if diff_path.exists():
            accepted[str(patch["cluster_id"])] = diff_path.read_text(encoding="utf-8")[:4000]
    return accepted


def _coverage_sentence(
    *,
    verdict: str,
    n: int,
    pct: float,
    ci: dict[str, float],
    degraded: bool,
) -> str:
    # SPEC-NOTE: SPEC §7 originally pinned a sentence containing the banned term
    # "guarantee"; M6's certificate language gate correctly rejects that term.
    # M6.5 locks this replacement as the canonical contract sentence.
    base = (
        f"{verdict} at n={n}, covering {pct:.1f}% of production template distribution, "
        f"95% CI {_coverage_ci_text(ci)}. "
        "This certifies the sampled workload against the stated assertions; "
        "it does not certify unsampled inputs."
    )
    if degraded:
        return (
            f"{base} Noise floor unavailable (degraded mode): "
            "behavioral-variance comparison was not performed."
        )
    return base


def _coverage_ci_text(ci: dict[str, float]) -> str:
    return f"[{_format_ci_percent(ci['lower'])}, {_format_ci_percent(ci['upper'])}]"


def _format_ci_percent(value: float) -> str:
    percent = value * 100
    digits = 2 if 0 < abs(percent) < 0.1 else 1
    return f"{percent:.{digits}f}%"


def _coverage_percent(sampling_report: dict[str, Any], traces: list[TraceRecord]) -> float:
    if "production_template_coverage_pct" in sampling_report:
        return float(sampling_report["production_template_coverage_pct"])
    sampled_template_ids = {trace.meta.template_id for trace in traces if trace.meta.template_id}
    if not sampled_template_ids:
        return 100.0
    return 100.0


def _exclusion_percent(sampling_report: dict[str, Any]) -> float:
    value = sampling_report.get("exclusion_percent", 0.0)
    return round(float(value), 12)


def _dedup_scope(sampling_report: dict[str, Any]) -> dict[str, Any]:
    # Disclose what near-duplicate collapse removed before sampling. Without this,
    # an "exhaustive, 0.0% excluded" line hides that N collected traces were
    # collapsed into representatives (run #1 collapsed 21 silently).
    dedup = sampling_report.get("dedup") or {}
    collapsed = int(dedup.get("collapsed_count", 0))
    n_after_dedup = int(sampling_report.get("n_available_after_dedup", 0))
    return {
        "dedup_collapsed": collapsed,
        "dedup_method": {
            "jaccard_threshold": dedup.get("jaccard_threshold"),
            "shingle_size": dedup.get("shingle_size"),
        },
        "n_after_dedup": n_after_dedup,
        "n_collected": n_after_dedup + collapsed,
    }


def _zero_event_sentence(n: int) -> str:
    return (
        f"0 critical failures observed at n={n}; 95% upper bound on the true critical rate "
        f"\u2264 {rule_of_three_upper_bound(n) * 100:.1f}%."
    )


def _judge_health(judge_stats: dict[str, Any]) -> dict[str, Any]:
    """Panel-wide reliability so a collapsed judge layer is visible on the cert.

    Every (pair, judge) evaluation either yields a valid verdict or abstains
    (parse failure, position inconsistency, or call error). Run 0a716021's panel
    produced 100 valid verdicts out of 426 evaluations — a collapse the headline
    hid; these rates put it on the face of the certificate.
    """
    judges = judge_stats.get("judges") or {}
    valid = abstentions = parse_failures = inconsistencies = call_errors = 0
    for judge in judges.values():
        valid += sum(int(count) for count in (judge.get("verdict_counts") or {}).values())
        abstentions += int(judge.get("abstentions", 0))
        parse_failures += int(judge.get("parse_failures", 0))
        inconsistencies += int(judge.get("position_inconsistencies", 0))
        call_errors += int(judge.get("call_errors", 0))
    evaluations = valid + abstentions

    def _rate(value: int) -> float | None:
        return round(value / evaluations, 12) if evaluations else None

    return {
        "abstentions": abstentions,
        "call_errors": call_errors,
        "evaluations": evaluations,
        "parse_failure_rate": _rate(parse_failures),
        "parse_failures": parse_failures,
        "position_inconsistencies": inconsistencies,
        "position_inconsistency_rate": _rate(inconsistencies),
        "valid_verdict_rate": _rate(valid),
        "valid_verdicts": valid,
    }


def _judge_panel(judge_stats: dict[str, Any], *, judged: bool) -> dict[str, Any]:
    judges = sorted((judge_stats.get("judges") or {}).keys())
    ensemble = judge_stats.get("ensemble") or {}
    calibration = judge_stats.get("calibration") or {}
    run = judge_stats.get("run") or {}
    completed_pairs = int(run.get("completed_pairs") or 0)
    judge_calls = int(run.get("dispatched") or 0) + int(run.get("cache_hits") or 0)
    # SPEC-NOTE: run #1 rendered "n/a — no pairs required judging" while 352
    # judgment rows existed, because the reason was inferred from agreement being
    # None. Agreement/alpha are legitimately None when judging DID occur but no
    # pair collected ≥2 comparable verdicts (e.g. one judge abstained on every
    # pair). "Whether judging occurred" is now read from the presence of judgment
    # rows, never from the agreement value.
    agreement_percent = ensemble.get("raw_agreement_percent")
    krippendorff_alpha = ensemble.get("krippendorff_alpha")
    calibration_accuracy = calibration.get("accuracy")
    return {
        "agreement_percent": agreement_percent,
        "agreement_reason": _judge_unavailable_reason(
            value=agreement_percent,
            judged=judged,
            judges=judges,
        ),
        "calibration_accuracy": calibration_accuracy,
        "calibration_accuracy_reason": (
            None if calibration_accuracy is not None else "no calibration set provided"
        ),
        "completed_pairs": completed_pairs,
        "families": sorted({str(judge).split("/", 1)[0] for judge in judges}),
        "judge_calls": judge_calls,
        "krippendorff_alpha": krippendorff_alpha,
        "krippendorff_alpha_reason": _judge_unavailable_reason(
            value=krippendorff_alpha,
            judged=judged,
            judges=judges,
        ),
        "models": judges,
        "order_swap": True,
    }


def _judge_unavailable_reason(
    *,
    value: object,
    judged: bool,
    judges: list[str],
) -> str | None:
    if value is not None:
        return None
    if not judged:
        return "no pairs required judging"
    if len(judges) < 2:
        return "requires ≥2 judges with comparable verdicts"
    return "no pair had ≥2 comparable judge verdicts"


def _cost_latency_payload(
    baseline_calls: list[dict[str, Any]],
    candidate_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_tokens = _mean_output_tokens(baseline_calls)
    candidate_tokens = _mean_output_tokens(candidate_calls)
    # SPEC-NOTE: this ratio compares mean OUTPUT-token counts, so it measures
    # output-length inflation, not tokenizer inflation. It was previously keyed
    # "tokenizer_inflation", which was wrong: tokenizer inflation is the ratio of
    # two tokenizers over the SAME text, and computing it needs both providers'
    # tokenizers loaded — not available offline/cheaply here — so it is omitted
    # rather than approximated.
    return {
        "baseline_mean_latency_ms": _mean_latency(baseline_calls),
        "baseline_mean_output_tokens": baseline_tokens,
        "candidate_mean_latency_ms": _mean_latency(candidate_calls),
        "candidate_mean_output_tokens": candidate_tokens,
        "latency_delta_ms": _mean_latency(candidate_calls) - _mean_latency(baseline_calls),
        "output_length_inflation": candidate_tokens / baseline_tokens if baseline_tokens else None,
        "usd_delta": round(_total_cost(candidate_calls) - _total_cost(baseline_calls), 12),
    }


def _mean_output_tokens(calls: list[dict[str, Any]]) -> float:
    values = [
        float((call.get("usage") or {}).get("output_tokens", 0))
        for call in calls
    ]
    return round(sum(values) / len(values), 12) if values else 0.0


def _mean_latency(calls: list[dict[str, Any]]) -> float:
    values = [
        float((call.get("response") or {}).get("latency_ms", 0))
        for call in calls
    ]
    return round(sum(values) / len(values), 12) if values else 0.0


def _total_cost(calls: list[dict[str, Any]]) -> float:
    return round(sum(float(call.get("cost_usd", 0.0)) for call in calls), 12)


def _truncate_exemplar(value: str) -> str:
    return value if len(value) <= 300 else f"{value[:297]}..."


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing artifact: {paths[0]}")


def _read_json(path: Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = hashlib.sha256(f"{content}\n".encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _stable_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
