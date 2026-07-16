from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from acsi.cert.build import (
    BANNED_PHRASES,
    BannedLanguageError,
    assert_no_banned_language,
)

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
REPORT_TEMPLATE = "report.html.j2"
ALPINE_PATH = TEMPLATE_DIR / "alpine.min.js"


def assert_no_banned_words(rendered: str) -> None:
    assert_no_banned_language(rendered)


def render_report(
    cert: dict[str, Any],
    *,
    output_path: Path,
    template_dir: Path = TEMPLATE_DIR,
    review_mode: bool = False,
) -> str:
    rendered = render_report_html(cert, template_dir=template_dir, review_mode=review_mode)
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
    env.filters["criterion_input"] = _criterion_input
    env.filters["criterion_label"] = _criterion_label
    env.filters["criterion_status"] = _criterion_status
    env.filters["criterion_threshold"] = _criterion_threshold
    env.filters["decimal"] = _format_decimal
    env.filters["ms"] = _format_ms
    env.filters["multiplier"] = _format_multiplier
    env.filters["pct"] = _format_percent
    env.filters["pct_value"] = _format_percent_value
    env.filters["usd"] = _format_usd
    template = env.get_template(REPORT_TEMPLATE)
    certificate_json = json.dumps(
        _html_json_value(cert),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    rendered = template.render(
        alpine_js=alpine_source,
        banned_phrases=", ".join(BANNED_PHRASES),
        cert=cert,
        certificate_json=certificate_json,
        payload=cert["payload"],
        review_mode=review_mode,
    )
    assert_no_banned_language(rendered)
    return rendered


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


def _criterion_label(criterion: dict[str, Any]) -> str:
    labels = {
        "candidate_regression_rate": "judge-adjudicated regression vs noise floor",
        "critical_assertions": "critical assertion failures",
        "critical_cluster_share": "critical cluster share",
    }
    return labels.get(str(criterion.get("id")), str(criterion.get("id")))


def _criterion_input(criterion: dict[str, Any]) -> str:
    criterion_id = str(criterion.get("id"))
    if criterion_id == "critical_assertions":
        return f"{int(criterion.get('actual', 0))} failures"
    if criterion_id == "candidate_regression_rate":
        if criterion.get("mode") == "degraded":
            return f"n/a — {criterion.get('reason')}"
        return (
            f"candidate upper {_format_ci_percent(criterion.get('actual_ci_upper', 0.0))} "
            f"vs noise upper {_format_ci_percent(criterion.get('baseline_ci_upper', 0.0))} "
            f"+ epsilon {_format_percent(criterion.get('epsilon', 0.0))}"
        )
    if criterion_id == "critical_cluster_share":
        actual = criterion.get("actual") or []
        if not isinstance(actual, list) or not actual:
            return "no clusters above threshold"
        return ", ".join(
            f"{item.get('cluster_id')} at {_format_percent(item.get('share_of_sampled', 0.0))}"
            for item in actual
            if isinstance(item, dict)
        )
    return str(criterion)


def _criterion_threshold(criterion: dict[str, Any]) -> str:
    criterion_id = str(criterion.get("id"))
    if criterion_id == "critical_assertions":
        return str(int(criterion.get("threshold", 0)))
    if criterion.get("mode") == "degraded":
        return str(criterion.get("reason", "n/a"))
    if criterion_id in {"candidate_regression_rate", "critical_cluster_share"}:
        return _format_percent(criterion.get("threshold", 0.0))
    return str(criterion.get("threshold", criterion.get("reason", "n/a")))


def _criterion_status(criterion: dict[str, Any]) -> str:
    passed = criterion.get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "BLOCK"
    return "n/a"


__all__ = [
    "BannedLanguageError",
    "assert_no_banned_words",
    "render_report",
    "render_report_html",
]
