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
    template = env.get_template(REPORT_TEMPLATE)
    certificate_json = json.dumps(
        cert,
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
    )
    assert_no_banned_language(rendered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        if not rendered.endswith("\n"):
            handle.write("\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    with Path(f"{output_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


__all__ = [
    "BannedLanguageError",
    "assert_no_banned_words",
    "render_report",
]
