from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_REPOS = (
    "kubernetes/kubernetes",
    "microsoft/vscode",
    "facebook/react",
    "rust-lang/rust",
)
DEFAULT_OUTPUT = Path("benchmarks/oss-issues/corpus.jsonl")
GITHUB_API = "https://api.github.com"
MIN_BODY_CHARS = 200
MAX_BODY_CHARS = 4_000


@dataclass(frozen=True)
class CorpusItem:
    source_repo: str
    issue_number: int
    html_url: str
    title: str
    body: str
    fetched_at: str
    truncated: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "body": self.body,
            "fetched_at": self.fetched_at,
            "html_url": self.html_url,
            "issue_number": self.issue_number,
            "source_repo": self.source_repo,
            "title": self.title,
            "truncated": self.truncated,
        }


@dataclass
class CorpusBuildStats:
    fetched: int = 0
    pages: int = 0
    accepted: int = 0
    emitted: int = 0
    truncated: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)
    emitted_by_repo: Counter[str] = field(default_factory=Counter)

    def to_payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "emitted": self.emitted,
            "emitted_by_repo": dict(sorted(self.emitted_by_repo.items())),
            "exclusions": dict(sorted(self.exclusions.items())),
            "fetched": self.fetched,
            "pages": self.pages,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class CorpusBuildResult:
    items: list[CorpusItem]
    output_path: Path
    stats: CorpusBuildStats


def build_corpus(
    *,
    repos: list[str],
    n: int,
    output_path: Path,
    token: str,
    client: httpx.Client | None = None,
    fetched_at: str | None = None,
    api_url: str = GITHUB_API,
) -> CorpusBuildResult:
    if n <= 0:
        raise ValueError("--n must be positive.")
    if not token:
        raise ValueError("Set GITHUB_TOKEN to fetch GitHub issues before building the corpus.")

    timestamp = fetched_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stats = CorpusBuildStats()
    accepted_by_repo: dict[str, list[CorpusItem]] = {}
    seen: set[tuple[str, int]] = set()
    close_client = client is None
    active_client = client or httpx.Client(timeout=30.0)
    try:
        for repo in repos:
            accepted_by_repo[repo] = _fetch_repo_items(
                repo=repo,
                n=n,
                token=token,
                client=active_client,
                fetched_at=timestamp,
                api_url=api_url,
                seen=seen,
                stats=stats,
            )
    finally:
        if close_client:
            active_client.close()

    items = _round_robin(accepted_by_repo, n)
    stats.emitted = len(items)
    stats.emitted_by_repo = Counter(item.source_repo for item in items)
    _write_jsonl(output_path, [item.to_payload() for item in items])
    return CorpusBuildResult(items=items, output_path=output_path, stats=stats)


def _fetch_repo_items(
    *,
    repo: str,
    n: int,
    token: str,
    client: httpx.Client,
    fetched_at: str,
    api_url: str,
    seen: set[tuple[str, int]],
    stats: CorpusBuildStats,
) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    page = 1
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    while len(items) < n:
        response = client.get(
            f"{api_url}/repos/{repo}/issues",
            headers=headers,
            params={"page": page, "per_page": 100, "state": "closed"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"GitHub issues response for {repo} page {page} was not a list.")
        stats.pages += 1
        if not payload:
            break
        for raw_item in payload:
            stats.fetched += 1
            item = _normalize_item(repo, raw_item, fetched_at, seen, stats)
            if item is not None:
                items.append(item)
            if len(items) >= n:
                break
        page += 1
    return items


def _normalize_item(
    repo: str,
    raw_item: Any,
    fetched_at: str,
    seen: set[tuple[str, int]],
    stats: CorpusBuildStats,
) -> CorpusItem | None:
    if not isinstance(raw_item, dict):
        stats.exclusions["invalid_item"] += 1
        return None
    if "pull_request" in raw_item:
        stats.exclusions["pull_request"] += 1
        return None
    user = raw_item.get("user")
    if isinstance(user, dict) and user.get("type") == "Bot":
        stats.exclusions["bot"] += 1
        return None

    body = raw_item.get("body")
    if not isinstance(body, str) or len(body) < MIN_BODY_CHARS:
        stats.exclusions["short_body"] += 1
        return None

    number = raw_item.get("number")
    if not isinstance(number, int):
        stats.exclusions["invalid_item"] += 1
        return None
    dedupe_key = (repo, number)
    if dedupe_key in seen:
        stats.exclusions["duplicate"] += 1
        return None
    seen.add(dedupe_key)

    truncated = len(body) > MAX_BODY_CHARS
    if truncated:
        body = body[:MAX_BODY_CHARS]
        stats.truncated += 1
    stats.accepted += 1
    return CorpusItem(
        source_repo=repo,
        issue_number=number,
        html_url=str(raw_item.get("html_url") or ""),
        title=str(raw_item.get("title") or ""),
        body=body,
        fetched_at=fetched_at,
        truncated=truncated,
    )


def _round_robin(items_by_repo: dict[str, list[CorpusItem]], n: int) -> list[CorpusItem]:
    selected: list[CorpusItem] = []
    offsets: defaultdict[str, int] = defaultdict(int)
    repos = list(items_by_repo)
    while len(selected) < n:
        added = False
        for repo in repos:
            offset = offsets[repo]
            if offset >= len(items_by_repo[repo]):
                continue
            selected.append(items_by_repo[repo][offset])
            offsets[repo] += 1
            added = True
            if len(selected) >= n:
                break
        if not added:
            break
    return selected


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the OSS issue benchmark corpus.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="GitHub repo in owner/name form. May be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    try:
        result = build_corpus(
            repos=args.repos or list(DEFAULT_REPOS),
            n=args.n,
            output_path=args.output,
            token=token,
        )
    except (httpx.HTTPError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result.stats.to_payload(), sort_keys=True))


if __name__ == "__main__":
    main()
