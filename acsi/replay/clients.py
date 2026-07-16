from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionRequest:
    provider: str
    model: str
    prompt: str
    system: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResponse:
    text: str
    served_model: str
    usage: dict[str, int]
    finish_reason: str = "stop"
    latency_ms: int = 0
    tool_calls: list[dict[str, Any]] | None = None


class CompletionClient(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


class FakeClient:
    def __init__(self, noise: float = 0.0, regressions: dict[str, str] | None = None) -> None:
        self.noise = noise
        self.regressions = regressions or {}

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        for marker, replacement in self.regressions.items():
            if marker in request.prompt:
                return self._response(request, replacement)

        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        rng = random.Random(digest)
        variant = "summary"
        if rng.random() < self.noise:
            variant = "paraphrased summary"
        text = f"{variant}: {digest[:16]}"
        return self._response(request, text)

    @staticmethod
    def _response(request: CompletionRequest, text: str) -> CompletionResponse:
        return CompletionResponse(
            text=text,
            served_model=request.model,
            usage={"input_tokens": len(request.prompt.split()), "output_tokens": len(text.split())},
        )


class LiveClient:
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        _ = request
        raise NotImplementedError("Live provider replay is scheduled for M2.")

