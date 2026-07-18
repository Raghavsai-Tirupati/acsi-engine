from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal

from acsi.diff.deterministic import DiffResponse
from acsi.diff.semantic import EmbeddingClient, FakeEmbedder, classify_pair
from acsi.judge.clients import FakeJudge, judge_family
from acsi.judge.ensemble import (
    JudgeAccumulator,
    append_vote,
    grouped_votes,
    summarize_judge_stats,
)
from acsi.judge.rubric import (
    CandidateOutcome,
    JudgeParseError,
    map_position_verdict,
    parse_classifier_judgment,
    parse_pairwise_judgment,
    render_classifier_rubric,
    render_pairwise_rubric,
)
from acsi.replay.clients import (
    CompletionClient,
    CompletionRequest,
    CompletionResponse,
    PermanentError,
    ReplayClientError,
)
from acsi.replay.runner import ReplayAbortError, estimate_call_cost_usd
from acsi.replay.store import ReplayStore, StoredCall
from acsi.schemas import TraceRecord

JUDGE_PHASE = "judge"


class JudgeInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidatePair:
    pair_id: str
    trace_id: str
    prompt: str
    baseline: DiffResponse
    candidate: DiffResponse
    deterministic_equal: bool
    similarity: float


@dataclass(frozen=True)
class JudgeCallResult:
    parsed: object | None
    parse_failed: bool
    cache_hit: bool
    dispatched: bool
    cost_usd: float
    call_error: str | None = None


@dataclass(frozen=True)
class JudgeRunConfig:
    run_id: str
    seed: int = 42
    interrupt_after_dispatches: int | None = None
    call_timeout_s: float = 120.0
    max_attempts: int = 4
    base_backoff_s: float = 2.0
    max_retry_after_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    progress: Callable[[str], None] | None = None


@dataclass
class JudgeRunResult:
    judgments: list[dict[str, object]]
    stats: dict[str, object]
    completed_pairs: int
    dispatched: int = 0
    cache_hits: int = 0
    parse_failures: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ClassifierAssertionResult:
    assertion_id: str
    pair_id: str
    passed: bool
    queued_for_review: bool
    votes: dict[str, bool | None]
    parse_failures: int


@dataclass
class DispatchBudget:
    interrupt_after_dispatches: int | None
    dispatched: int = 0

    def record_dispatch(self) -> None:
        self.dispatched += 1
        if (
            self.interrupt_after_dispatches is not None
            and self.dispatched >= self.interrupt_after_dispatches
        ):
            raise JudgeInterrupted("Judge run interrupted; rerun to resume from checkpoints.")


def build_candidate_pairs(
    traces: list[TraceRecord],
    baseline_calls: list[StoredCall],
    candidate_calls: list[StoredCall],
    *,
    tau: float,
    embedder: EmbeddingClient | None = None,
) -> list[CandidatePair]:
    baseline_by_trace = _calls_by_trace_sample(baseline_calls, sample_index=0)
    candidate_by_trace = _calls_by_trace_sample(candidate_calls, sample_index=0)
    active_embedder = embedder or FakeEmbedder()
    pairs: list[CandidatePair] = []
    for trace in sorted(traces, key=lambda record: str(record.trace_id)):
        trace_id = str(trace.trace_id)
        if trace_id not in baseline_by_trace or trace_id not in candidate_by_trace:
            continue
        baseline = _diff_response(baseline_by_trace[trace_id])
        candidate = _diff_response(candidate_by_trace[trace_id])
        classification = classify_pair(
            baseline,
            candidate,
            embedder=active_embedder,
            threshold=tau,
        )
        pairs.append(
            CandidatePair(
                pair_id=trace_id,
                trace_id=trace_id,
                prompt=trace.request.messages[0].content,
                baseline=baseline,
                candidate=candidate,
                deterministic_equal=classification.deterministic_equal,
                similarity=classification.similarity,
            )
        )
    return pairs


def select_for_judging(pairs: list[CandidatePair], tau: float) -> list[CandidatePair]:
    return [
        pair
        for pair in pairs
        if not pair.deterministic_equal and pair.similarity < tau
    ]


def run_pairwise_judging(
    pairs: list[CandidatePair],
    judge_clients: dict[str, CompletionClient],
    *,
    store: ReplayStore,
    config: JudgeRunConfig,
) -> JudgeRunResult:
    store.initialize()
    accumulators = {
        judge: JudgeAccumulator() for judge in sorted(judge_clients)
    }
    votes_by_pair = grouped_votes()
    judgments: list[dict[str, object]] = []
    budget = DispatchBudget(config.interrupt_after_dispatches)
    cache_hits = 0
    cost_usd = 0.0
    call_errors = 0

    ordered_pairs = sorted(pairs, key=lambda item: item.pair_id)
    total = len(ordered_pairs)
    for index, pair in enumerate(ordered_pairs, start=1):
        for judge in sorted(judge_clients):
            client = judge_clients[judge]
            accumulator = accumulators[judge]
            accumulator.pairs_seen += 1

            def report(attempt: int, judge: str = judge, index: int = index) -> None:
                _emit_progress(config, f"judging pair {index}/{total} [{judge}] attempt {attempt}")

            left = _judge_pair_ordering(
                pair,
                judge,
                client,
                store=store,
                config=config,
                ordering="candidate_a",
                budget=budget,
                on_attempt=report,
            )
            right = _judge_pair_ordering(
                pair,
                judge,
                client,
                store=store,
                config=config,
                ordering="candidate_b",
                budget=budget,
                on_attempt=report,
            )
            cache_hits += int(left.cache_hit) + int(right.cache_hit)
            cost_usd += left.cost_usd + right.cost_usd
            outcome, reason = _combine_orderings(left, right)
            if reason == "judge_error":
                # SPEC-NOTE: a judge whose call exhausts transient retries abstains
                # (records a judge_error row) instead of aborting the whole run.
                # Its vote is None; downstream min_judges reclassifies pairs that
                # then fall below the panel floor to "unresolved".
                accumulator.call_errors += 1
                accumulator.abstentions += 1
                call_errors += 1
            elif reason == "parse_failure":
                accumulator.parse_failures += 1
                accumulator.abstentions += 1
            elif reason == "position_inconsistency":
                accumulator.position_inconsistencies += 1
                accumulator.abstentions += 1
            elif outcome is not None:
                accumulator.verdict_counts[outcome] += 1
            else:
                accumulator.abstentions += 1
            append_vote(votes_by_pair, pair.pair_id, judge, outcome)
            judgments.append(
                {
                    "abstain_reason": reason,
                    "error": left.call_error or right.call_error,
                    "judge": judge,
                    "outcome": outcome,
                    "pair_id": pair.pair_id,
                }
            )

    stats = summarize_judge_stats(accumulators, votes_by_pair)
    _emit_progress(
        config,
        f"judged {total} pairs; {budget.dispatched} calls, {call_errors} judge errors, "
        f"{sum(acc.parse_failures for acc in accumulators.values())} parse failures",
    )
    return JudgeRunResult(
        judgments=judgments,
        stats=stats,
        completed_pairs=len(pairs),
        dispatched=budget.dispatched,
        cache_hits=cache_hits,
        parse_failures=sum(acc.parse_failures for acc in accumulators.values()),
        cost_usd=cost_usd,
    )


def _emit_progress(config: JudgeRunConfig, message: str) -> None:
    if config.progress is not None:
        config.progress(message)


def run_classifier_assertion(
    *,
    assertion_id: str,
    pair_id: str,
    prompt: str,
    response: str,
    criterion: str,
    judge_clients: dict[str, CompletionClient],
    store: ReplayStore,
    config: JudgeRunConfig,
) -> ClassifierAssertionResult:
    store.initialize()
    votes: dict[str, bool | None] = {}
    parse_failures = 0
    budget = DispatchBudget(config.interrupt_after_dispatches)
    for judge in sorted(judge_clients):
        prompt_text = render_classifier_rubric(prompt, response, criterion)
        call = _call_with_parse_retry(
            judge,
            judge_clients[judge],
            prompt_text,
            store=store,
            config=config,
            call_id=f"classifier:{assertion_id}:{pair_id}:{judge}",
            params={
                "assertion_id": assertion_id,
                "mode": "classifier",
                "ordering": "single",
                "pair_id": pair_id,
            },
            parser=parse_classifier_judgment,
            budget=budget,
        )
        if call.parse_failed or call.parsed is None:
            votes[judge] = None
            parse_failures += 1
        else:
            votes[judge] = bool(call.parsed.passed)
    pass_votes = sum(1 for vote in votes.values() if vote is True)
    fail_votes = sum(1 for vote in votes.values() if vote is False)
    passed = pass_votes > fail_votes
    queued = pass_votes == fail_votes
    return ClassifierAssertionResult(
        assertion_id=assertion_id,
        pair_id=pair_id,
        passed=passed,
        queued_for_review=queued,
        votes=votes,
        parse_failures=parse_failures,
    )


def write_judge_artifacts(
    run_dir: Path,
    result: JudgeRunResult,
    *,
    calibration: dict[str, object] | None = None,
) -> tuple[str, str]:
    judgments_hash = _write_jsonl(
        run_dir / "judgments.jsonl",
        result.judgments,
    )
    stats = dict(result.stats)
    stats["run"] = {
        "cache_hits": result.cache_hits,
        "completed_pairs": result.completed_pairs,
        "cost_ledger": [
            {
                "stage": "judge",
                "tokens_in": None,
                "tokens_out": None,
                "usd": round(result.cost_usd, 12),
            }
        ],
        "dispatched": result.dispatched,
        "parse_failures": result.parse_failures,
    }
    if calibration is not None:
        stats["calibration"] = calibration
    stats_hash = _write_json(run_dir / "judge_stats.json", stats)
    return judgments_hash, stats_hash


def _judge_pair_ordering(
    pair: CandidatePair,
    judge: str,
    client: CompletionClient,
    *,
    store: ReplayStore,
    config: JudgeRunConfig,
    ordering: Literal["candidate_a", "candidate_b"],
    budget: DispatchBudget,
    on_attempt: Callable[[int], None] | None = None,
) -> JudgeCallResult:
    if ordering == "candidate_a":
        response_a = pair.candidate.text or ""
        response_b = pair.baseline.text or ""
    else:
        response_a = pair.baseline.text or ""
        response_b = pair.candidate.text or ""
    prompt_text = render_pairwise_rubric(pair.prompt, response_a, response_b)
    return _call_with_parse_retry(
        judge,
        client,
        prompt_text,
        store=store,
        config=config,
        call_id=f"pairwise:{pair.pair_id}:{judge}:{ordering}",
        params={
            "mode": "pairwise",
            "ordering": ordering,
            "pair_id": pair.pair_id,
        },
        parser=parse_pairwise_judgment,
        budget=budget,
        on_attempt=on_attempt,
    )


def _call_with_parse_retry(
    judge: str,
    client: CompletionClient,
    prompt_text: str,
    *,
    store: ReplayStore,
    config: JudgeRunConfig,
    call_id: str,
    params: dict[str, object],
    parser,
    budget: DispatchBudget,
    on_attempt: Callable[[int], None] | None = None,
) -> JudgeCallResult:
    cache_hit = False
    cost_usd = 0.0
    for attempt in range(2):
        cached = store.get_done(
            config.run_id,
            call_id,
            attempt,
            phase=JUDGE_PHASE,
        )
        if cached:
            cache_hit = True
            response_text = cached.response.get("text") if cached.response else None
            cost_usd += cached.cost_usd
        else:
            request = CompletionRequest(
                provider=judge_family(judge),
                model=judge,
                system=None,
                messages=[{"role": "user", "content": prompt_text}],
                params={**params, "attempt": attempt},
                sample_index=attempt,
            )
            try:
                response = _complete_with_transient_retry(
                    client,
                    request,
                    config,
                    on_attempt=on_attempt,
                )
            except PermanentError as exc:
                if exc.run_level:
                    # Provider-fatal (billing/quota/auth): abort the run rather
                    # than silently abstaining every remaining judge call.
                    raise ReplayAbortError(str(exc), status_code=exc.status_code) from exc
                return _call_error_result(cache_hit, cost_usd, exc)
            except ReplayClientError as exc:
                # Transient retries exhausted or another non-fatal provider error:
                # abstain for this call rather than crashing the run.
                return _call_error_result(cache_hit, cost_usd, exc)
            actual_cost = _judge_call_cost(request, response, client)
            store.write_done(
                run_id=config.run_id,
                trace_id=call_id,
                sample_index=attempt,
                model=judge,
                params_hash=_params_hash(request.params),
                prompt_hash=_prompt_hash(prompt_text),
                response=response,
                cost_usd=actual_cost,
                retry_count=0,
                phase=JUDGE_PHASE,
            )
            budget.record_dispatch()
            response_text = response.text
            cost_usd += actual_cost
        try:
            return JudgeCallResult(
                parsed=parser(response_text),
                parse_failed=False,
                cache_hit=cache_hit,
                dispatched=not cache_hit,
                cost_usd=cost_usd,
            )
        except JudgeParseError:
            continue
    return JudgeCallResult(
        parsed=None,
        parse_failed=True,
        cache_hit=cache_hit,
        dispatched=not cache_hit,
        cost_usd=cost_usd,
    )


def _call_error_result(cache_hit: bool, cost_usd: float, exc: Exception) -> JudgeCallResult:
    return JudgeCallResult(
        parsed=None,
        parse_failed=False,
        cache_hit=cache_hit,
        dispatched=True,
        cost_usd=cost_usd,
        call_error=str(exc),
    )


def _complete_with_transient_retry(
    client: CompletionClient,
    request: CompletionRequest,
    config: JudgeRunConfig,
    *,
    on_attempt: Callable[[int], None] | None = None,
) -> CompletionResponse:
    """Call the judge client, retrying only transient failures (429/5xx/timeout).

    Provider retry-after hints are honored up to max_retry_after_s; otherwise
    exponential backoff with jitter. Non-transient errors (400/401/404) raise on
    the first attempt.
    """
    last_error: ReplayClientError | None = None
    for attempt in range(1, config.max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt)
        try:
            return client.complete(request)
        except ReplayClientError as exc:
            last_error = exc
            if not getattr(exc, "retryable", False) or attempt >= config.max_attempts:
                raise
            config.sleep(_retry_delay_seconds(exc, config, attempt))
    assert last_error is not None
    raise last_error


def _retry_delay_seconds(
    exc: ReplayClientError,
    config: JudgeRunConfig,
    attempt: int,
) -> float:
    hint = getattr(exc, "retry_after_s", None)
    if hint is not None:
        return min(float(hint), config.max_retry_after_s)
    jitter = Random(f"{config.seed}:{attempt}").uniform(0, config.base_backoff_s)
    return config.base_backoff_s * (2 ** (attempt - 1)) + jitter


def _combine_orderings(
    left: JudgeCallResult,
    right: JudgeCallResult,
) -> tuple[CandidateOutcome | None, str | None]:
    if left.call_error or right.call_error:
        return None, "judge_error"
    if left.parse_failed or right.parse_failed or left.parsed is None or right.parsed is None:
        return None, "parse_failure"
    left_outcome = map_position_verdict(left.parsed, candidate_position="a")
    right_outcome = map_position_verdict(right.parsed, candidate_position="b")
    if left_outcome != right_outcome:
        return None, "position_inconsistency"
    return left_outcome, None


def _calls_by_trace_sample(calls: list[StoredCall], *, sample_index: int) -> dict[str, StoredCall]:
    return {
        call.trace_id: call
        for call in calls
        if call.sample_index == sample_index
    }


def _diff_response(call: StoredCall) -> DiffResponse:
    response = call.response or {}
    return DiffResponse(
        text=response.get("text"),
        tool_calls=response.get("tool_calls"),
        finish_reason=response.get("finish_reason"),
        latency_ms=response.get("latency_ms"),
    )


def _judge_call_cost(
    request: CompletionRequest,
    response: CompletionResponse,
    client: CompletionClient,
) -> float:
    return estimate_call_cost_usd(
        request.provider,
        request.model,
        response.usage.get("input_tokens", 0),
        response.usage.get("output_tokens", 0),
        fake=isinstance(client, FakeJudge),
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{json.dumps(row, sort_keys=True, separators=(',', ':'))}\n"
        for row in sorted(rows, key=lambda item: (str(item["pair_id"]), str(item["judge"])))
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return _write_sha256(path, content)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    return _write_sha256(path, f"{content}\n")


def _write_sha256(path: Path, content: str) -> str:
    digest = hashlib.sha256(content.encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _params_hash(params: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
