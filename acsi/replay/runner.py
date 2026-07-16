from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acsi.replay.clients import (
    CompletionClient,
    CompletionRequest,
    CompletionResponse,
    FakeClient,
    PermanentError,
    RateLimitError,
    TransientError,
    plausible_token_count,
    prompt_sha256,
)
from acsi.replay.params import AppliedParamTransform, transform_params
from acsi.replay.store import ReplayStore, StoredCall
from acsi.schemas import ProviderModel, TraceRecord


@dataclass(frozen=True)
class ProviderRateLimit:
    requests_per_minute: int = 600
    tokens_per_minute: int = 1_000_000


@dataclass(frozen=True)
class ReplayConfig:
    run_id: str
    seed: int = 42
    concurrency: int = 4
    max_attempts: int = 5
    base_backoff_seconds: float = 0.01
    max_cost_usd: float | None = None
    provider_limits: dict[str, ProviderRateLimit] = field(default_factory=dict)
    interrupt_after_dispatches: int | None = None
    resume_command: str | None = None


@dataclass
class ReplayResult:
    run_id: str
    completed: int = 0
    errors: int = 0
    cache_hits: int = 0
    dispatched: int = 0
    retry_count: int = 0
    cost_usd: float = 0.0
    halted_reason: str | None = None
    param_transforms: list[AppliedParamTransform] = field(default_factory=list)
    served_models: set[str] = field(default_factory=set)


class ReplayAbortError(RuntimeError):
    pass


class ReplayInterrupted(RuntimeError):
    def __init__(self, resume_command: str | None) -> None:
        message = "Replay interrupted; rerun to resume from the checkpoint."
        if resume_command:
            message = f"{message} Resume command: {resume_command}"
        super().__init__(message)
        self.resume_command = resume_command


class BudgetExceeded(RuntimeError):
    pass


class AsyncTokenBucket:
    def __init__(self, *, per_minute: int) -> None:
        self.capacity = max(1, float(per_minute))
        self.tokens = self.capacity
        self.refill_per_second = self.capacity / 60
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: int = 1) -> None:
        amount = max(1, amount)
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
                self.updated_at = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                missing = amount - self.tokens
                wait_seconds = missing / self.refill_per_second
            await asyncio.sleep(wait_seconds)


class BudgetTracker:
    def __init__(self, max_cost_usd: float | None, spent_usd: float = 0.0) -> None:
        self.max_cost_usd = max_cost_usd
        self.spent_usd = spent_usd
        self.reserved_usd = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self, estimate_usd: float) -> None:
        async with self.lock:
            if self.max_cost_usd is not None:
                projected = self.spent_usd + self.reserved_usd + estimate_usd
                if projected > self.max_cost_usd:
                    raise BudgetExceeded
            self.reserved_usd += estimate_usd

    async def settle(self, estimate_usd: float, actual_usd: float) -> None:
        async with self.lock:
            self.reserved_usd = max(0.0, self.reserved_usd - estimate_usd)
            self.spent_usd += actual_usd


async def replay(
    traces: list[TraceRecord],
    model: ProviderModel,
    k_samples: int,
    *,
    client: CompletionClient,
    store: ReplayStore,
    config: ReplayConfig,
) -> ReplayResult:
    store.initialize()
    result = ReplayResult(run_id=config.run_id, cost_usd=store.total_cost(config.run_id))
    budget = BudgetTracker(config.max_cost_usd, spent_usd=result.cost_usd)
    semaphore = asyncio.Semaphore(config.concurrency)
    provider_buckets = _provider_buckets(config)
    halt_event = asyncio.Event()
    dispatch_lock = asyncio.Lock()
    dispatch_count = 0

    async def run_one(trace: TraceRecord, sample_index: int) -> None:
        nonlocal dispatch_count
        if halt_event.is_set():
            return

        trace_id = str(trace.trace_id)
        cached = store.get_done(config.run_id, trace_id, sample_index)
        if cached:
            result.cache_hits += 1
            result.completed += 1
            if cached.served_model:
                result.served_models.add(cached.served_model)
            return

        async with semaphore:
            if halt_event.is_set():
                return
            request, prompt_hash, params_hash, transforms = build_completion_request(
                trace,
                model,
                sample_index,
            )
            result.param_transforms.extend(transforms)
            estimate_usd = estimate_call_cost_usd(
                request.provider,
                request.model,
                plausible_token_count(request.prompt_text),
                estimated_output_tokens(trace),
                fake=isinstance(client, FakeClient),
            )
            try:
                await budget.reserve(estimate_usd)
            except BudgetExceeded:
                result.halted_reason = _budget_message(config.max_cost_usd)
                halt_event.set()
                return

            await _acquire_provider_limits(provider_buckets[request.provider], request)
            response: CompletionResponse | None = None
            actual_cost = 0.0
            retry_count = 0
            try:
                response, retry_count = await _complete_with_retries(
                    client,
                    request,
                    config,
                )
                actual_cost = response_cost_usd(
                    request,
                    response,
                    fake=isinstance(client, FakeClient),
                )
                store.write_done(
                    run_id=config.run_id,
                    trace_id=trace_id,
                    sample_index=sample_index,
                    model=request.model,
                    params_hash=params_hash,
                    prompt_hash=prompt_hash,
                    response=response,
                    cost_usd=actual_cost,
                    retry_count=retry_count,
                )
                result.completed += 1
                result.dispatched += 1
                result.retry_count += retry_count
                result.served_models.add(response.served_model)
                async with dispatch_lock:
                    dispatch_count += 1
                    if (
                        config.interrupt_after_dispatches is not None
                        and dispatch_count >= config.interrupt_after_dispatches
                    ):
                        halt_event.set()
                        raise ReplayInterrupted(config.resume_command)
            except PermanentError as exc:
                if exc.run_level:
                    halt_event.set()
                    raise ReplayAbortError(str(exc)) from exc
                store.write_error(
                    run_id=config.run_id,
                    trace_id=trace_id,
                    sample_index=sample_index,
                    model=request.model,
                    params_hash=params_hash,
                    prompt_hash=prompt_hash,
                    error=str(exc),
                    retry_count=retry_count,
                )
                result.errors += 1
            finally:
                await budget.settle(estimate_usd, actual_cost)
                result.cost_usd = budget.spent_usd

    tasks = [
        asyncio.create_task(run_one(trace, sample_index))
        for trace in traces
        for sample_index in range(k_samples)
    ]
    try:
        for task in asyncio.as_completed(tasks):
            await task
            if halt_event.is_set():
                break
    except (ReplayAbortError, ReplayInterrupted):
        _cancel_pending(tasks)
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if halt_event.is_set():
        _cancel_pending(tasks)
    await asyncio.gather(*tasks, return_exceptions=True)

    return result


def build_completion_request(
    trace: TraceRecord,
    model: ProviderModel,
    sample_index: int,
) -> tuple[CompletionRequest, str, str, list[AppliedParamTransform]]:
    mapped_params, transforms = transform_params(model.provider, model.model, trace.request.params)
    messages = [message.model_dump(mode="json") for message in trace.request.messages]
    request = CompletionRequest(
        provider=model.provider,
        model=model.model,
        system=trace.request.system,
        messages=messages,
        params=mapped_params,
        sample_index=sample_index,
    )
    return request, prompt_sha256(request.prompt_text), params_sha256(mapped_params), transforms


def params_sha256(params: dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def estimated_output_tokens(trace: TraceRecord) -> int:
    if trace.response.usage:
        return max(1, int(trace.response.usage.output_tokens * 1.2))
    return 128


def response_cost_usd(
    request: CompletionRequest,
    response: CompletionResponse,
    *,
    fake: bool,
) -> float:
    return estimate_call_cost_usd(
        request.provider,
        request.model,
        response.usage.get("input_tokens", 0),
        response.usage.get("output_tokens", 0),
        fake=fake,
    )


def estimate_call_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    fake: bool,
) -> float:
    if fake:
        input_price, output_price = FAKE_PRICE_PER_TOKEN.get(provider, (0.00000001, 0.00000002))
        return input_tokens * input_price + output_tokens * output_price
    try:
        from litellm.cost_calculator import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=f"{provider}/{model}",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def write_responses_jsonl(store: ReplayStore, run_id: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_stored_call_json(call) for call in store.done_calls(run_id)]
    content = "".join(f"{line}\n" for line in lines)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    with Path(f"{output_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


FAKE_PRICE_PER_TOKEN: dict[str, tuple[float, float]] = {
    "anthropic": (0.00000001, 0.00000002),
    "openai": (0.00000001, 0.00000002),
    "google": (0.00000001, 0.00000002),
    "fake": (0.00000001, 0.00000002),
}


def _provider_buckets(
    config: ReplayConfig,
) -> defaultdict[str, tuple[AsyncTokenBucket, AsyncTokenBucket]]:
    buckets: defaultdict[str, tuple[AsyncTokenBucket, AsyncTokenBucket]] = defaultdict(
        lambda: (
            AsyncTokenBucket(per_minute=ProviderRateLimit().requests_per_minute),
            AsyncTokenBucket(per_minute=ProviderRateLimit().tokens_per_minute),
        )
    )
    for provider, limits in config.provider_limits.items():
        buckets[provider] = (
            AsyncTokenBucket(per_minute=limits.requests_per_minute),
            AsyncTokenBucket(per_minute=limits.tokens_per_minute),
        )
    return buckets


async def _acquire_provider_limits(
    buckets: tuple[AsyncTokenBucket, AsyncTokenBucket],
    request: CompletionRequest,
) -> None:
    request_bucket, token_bucket = buckets
    await request_bucket.acquire(1)
    await token_bucket.acquire(plausible_token_count(request.prompt_text))


async def _complete_with_retries(
    client: CompletionClient,
    request: CompletionRequest,
    config: ReplayConfig,
) -> tuple[CompletionResponse, int]:
    retry_count = 0
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await asyncio.to_thread(client.complete, request), retry_count
        except RateLimitError as exc:
            retry_count += 1
            if attempt >= config.max_attempts:
                raise PermanentError("Rate limit retries exhausted.", run_level=False) from exc
            await asyncio.sleep(_backoff_seconds(config, request, attempt))
        except TransientError as exc:
            retry_count += 1
            if attempt >= config.max_attempts:
                raise PermanentError("Transient retries exhausted.", run_level=False) from exc
            await asyncio.sleep(_backoff_seconds(config, request, attempt))
    raise PermanentError("Retries exhausted.", run_level=False)


def _backoff_seconds(config: ReplayConfig, request: CompletionRequest, attempt: int) -> float:
    if config.base_backoff_seconds <= 0:
        return 0.0
    seed = f"{config.seed}:{request.model}:{request.prompt_text}:{request.sample_index}:{attempt}"
    jitter = random.Random(seed).uniform(0, config.base_backoff_seconds)
    return config.base_backoff_seconds * (2 ** (attempt - 1)) + jitter


def _budget_message(max_cost_usd: float | None) -> str:
    return f"Replay halted at --max-cost {max_cost_usd:.6f}; rerun the same command to resume."


def _stored_call_json(call: StoredCall) -> str:
    payload = {
        "trace_id": call.trace_id,
        "sample_index": call.sample_index,
        "model": call.model,
        "status": call.status,
        "response": call.response,
        "usage": call.usage,
        "cost_usd": call.cost_usd,
        "served_model": call.served_model,
        "retry_count": call.retry_count,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _cancel_pending(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
