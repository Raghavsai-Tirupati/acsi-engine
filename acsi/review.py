from __future__ import annotations

import json
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from acsi.cert.render import render_report_html
from acsi.overrides import (
    OVERRIDABLE_OUTCOMES,
    aggregate_judgment_rows,
    append_override,
    read_jsonl,
    read_overrides,
)


class ReviewError(ValueError):
    pass


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        run_id: str,
        run_dir: Path,
        manifest_path: Path,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.run_id = run_id
        self.active_run_dir = run_dir / "runs" / run_id
        self.manifest_path = manifest_path


def create_review_server(
    *,
    run_id: str,
    run_dir: Path = Path(".acsi"),
    manifest_path: Path = Path("acsi.yaml"),
    host: str = "127.0.0.1",
    port: int = 0,
) -> ReviewHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ReviewError("Review server only binds to loopback addresses.")
    bind_host = "127.0.0.1" if host == "localhost" else host
    return ReviewHTTPServer(
        (bind_host, port),
        ReviewHandler,
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )


def serve_review(
    *,
    run_id: str,
    run_dir: Path = Path(".acsi"),
    manifest_path: Path = Path("acsi.yaml"),
    port: int = 8765,
) -> None:
    server = create_review_server(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        port=port,
    )
    try:
        host, bound_port = server.server_address[:2]
        print(f"ACSI review server listening on http://{host}:{bound_port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self._send_html(_render_review_report(self.server.active_run_dir))
                return
            if path == "/api/run":
                self._send_json(_run_payload(self.server.active_run_dir))
                return
            self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
        except (OSError, json.JSONDecodeError, ReviewError) as exc:
            self._send_text(str(exc), status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/override":
                row = _handle_override(self.server.active_run_dir, self._read_body_json())
                self._send_json(row)
                return
            if path == "/api/promote-assertion":
                payload = _handle_promote_assertion(
                    self.server.manifest_path,
                    self._read_body_json(),
                )
                self._send_json(payload)
                return
            self._send_text("Not found", status=HTTPStatus.NOT_FOUND)
        except (OSError, json.JSONDecodeError, ReviewError, ValueError) as exc:
            self._send_text(str(exc), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, message: str, status: HTTPStatus) -> None:
        body = f"{message.strip()}\n".encode()
        self.send_response(status)
        self.send_header("content-type", "text/plain; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _render_review_report(active_run_dir: Path) -> str:
    cert_path = active_run_dir / "cert.json"
    if not cert_path.exists():
        raise ReviewError(f"Missing certificate at {cert_path}")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    return render_report_html(cert, review_mode=True)


def _run_payload(active_run_dir: Path) -> dict[str, Any]:
    judgment_rows = read_jsonl(active_run_dir / "judgments.jsonl")
    outcomes = aggregate_judgment_rows(judgment_rows)
    overrides = read_overrides(active_run_dir)
    unresolved = [
        {"outcome": outcome, "pair_id": pair_id}
        for pair_id, outcome in sorted(outcomes.items())
        if outcome == "unresolved"
    ]
    return {
        "outcomes": outcomes,
        "overrides": overrides,
        "run_id": active_run_dir.name,
        "unresolved_queue": unresolved,
    }


def _handle_override(active_run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    pair_id = str(payload.get("pair_id") or "").strip()
    to_outcome = str(payload.get("to_outcome") or "").strip()
    note = payload.get("note")
    if not pair_id:
        raise ReviewError("pair_id is required.")
    if to_outcome not in OVERRIDABLE_OUTCOMES:
        raise ReviewError("Override outcome must be a supported judge outcome.")

    judgment_rows = read_jsonl(active_run_dir / "judgments.jsonl")
    outcomes = aggregate_judgment_rows(judgment_rows)
    if pair_id not in outcomes:
        assertion_pairs = _assertion_pair_ids(active_run_dir / "assertion_results.jsonl")
        if pair_id in assertion_pairs:
            raise ReviewError("Assertion-derived outcomes are not overridable.")
        raise ReviewError("Only judge-derived ensemble outcomes are overridable.")
    row = append_override(
        active_run_dir,
        pair_id=pair_id,
        from_outcome=outcomes[pair_id],
        to_outcome=to_outcome,
        note=str(note) if note else None,
    )
    return row


def _assertion_pair_ids(path: Path) -> set[str]:
    return {
        str(row.get("pair_id") or row.get("trace_id"))
        for row in read_jsonl(path)
        if row.get("pair_id") or row.get("trace_id")
    }


def _handle_promote_assertion(manifest_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    assertion_type = str(payload.get("type") or "").strip()
    if assertion_type not in {"contains", "not_contains", "regex"}:
        raise ReviewError("Promoted assertion type must be contains, not_contains, or regex.")
    severity = str(payload.get("severity") or "major").strip()
    if severity not in {"critical", "major", "minor"}:
        raise ReviewError("Promoted assertion severity must be critical, major, or minor.")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ReviewError("Promoted assertion params must be an object.")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is present in supported envs.
        raise ReviewError("PyYAML is required for assertion promotion.") from exc

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    assertions = list(manifest.get("assertions") or [])
    assertion = {
        "id": str(payload.get("id") or f"promoted-{len(assertions) + 1}"),
        "severity": severity,
        "type": assertion_type,
        **params,
    }
    backup_path = Path(f"{manifest_path}.bak")
    shutil.copyfile(manifest_path, backup_path)
    # SPEC-NOTE: M7 promotion uses safe_load/safe_dump; ordering/comments are not preserved.
    manifest["assertions"] = assertions + [assertion]
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return {"assertion": assertion, "backup_path": str(backup_path), "status": "ok"}


__all__ = [
    "ReviewError",
    "create_review_server",
    "serve_review",
]
