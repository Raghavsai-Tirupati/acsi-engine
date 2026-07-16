from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts.build_oss_issue_corpus import build_corpus, main


def test_build_corpus_filters_truncates_paginates_and_balances(tmp_path: Path) -> None:
    requests: list[tuple[str, int, str]] = []
    transport = httpx.MockTransport(lambda request: _mock_github(request, requests))
    output = tmp_path / "corpus.jsonl"

    with httpx.Client(transport=transport) as client:
        result = build_corpus(
            repos=["owner/a", "owner/b", "owner/c"],
            n=7,
            output_path=output,
            token="test-token",
            client=client,
            fetched_at="2026-07-16T00:00:00Z",
        )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(row["source_repo"], row["issue_number"]) for row in rows] == [
        ("owner/a", 1),
        ("owner/b", 1),
        ("owner/c", 1),
        ("owner/a", 5),
        ("owner/b", 2),
        ("owner/c", 3),
        ("owner/a", 6),
    ]
    assert len(result.items) == 7
    assert result.stats.to_payload() == {
        "accepted": 7,
        "emitted": 7,
        "emitted_by_repo": {"owner/a": 3, "owner/b": 2, "owner/c": 2},
        "exclusions": {
            "bot": 1,
            "duplicate": 1,
            "pull_request": 1,
            "short_body": 2,
        },
        "fetched": 12,
        "pages": 7,
        "truncated": 1,
    }
    truncated = next(
        row for row in rows if row["source_repo"] == "owner/a" and row["issue_number"] == 5
    )
    assert truncated["truncated"] is True
    assert len(truncated["body"]) == 4000
    assert ("owner/a", 2, "Bearer test-token") in requests


def test_corpus_builder_cli_missing_token_is_one_line(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        main(["--output", str(tmp_path / "corpus.jsonl")])

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.strip() == (
        "Set GITHUB_TOKEN to fetch GitHub issues before building the corpus."
    )


def _mock_github(
    request: httpx.Request,
    requests: list[tuple[str, int, str]],
) -> httpx.Response:
    parts = request.url.path.strip("/").split("/")
    repo = "/".join(parts[1:3])
    page = int(request.url.params["page"])
    auth = request.headers.get("Authorization", "")
    requests.append((repo, page, auth))
    pages = {
        ("owner/a", 1): [
            _issue(1),
            _issue(2, pull_request=True),
            _issue(3, user_type="Bot"),
            _issue(4, body="s" * 199),
            _issue(5, body="L" * 4105),
        ],
        ("owner/a", 2): [_issue(6)],
        ("owner/a", 3): [],
        ("owner/b", 1): [_issue(1), _issue(2), _issue(2)],
        ("owner/b", 2): [],
        ("owner/c", 1): [_issue(1), _issue(2, body=None), _issue(3)],
        ("owner/c", 2): [],
    }
    return httpx.Response(200, json=pages[(repo, page)])


_DEFAULT_BODY = object()


def _issue(
    number: int,
    *,
    body: str | None | object = _DEFAULT_BODY,
    pull_request: bool = False,
    user_type: str = "User",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "body": f"Body for issue {number}. " * 20 if body is _DEFAULT_BODY else body,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "number": number,
        "title": f"Issue {number}",
        "user": {"type": user_type},
    }
    if pull_request:
        payload["pull_request"] = {"url": "https://api.github.com/pr"}
    return payload
