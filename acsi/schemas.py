from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TraceSource(StrEnum):
    CAPTURE = "capture"
    BACKFILL = "backfill"
    JSONL = "jsonl"
    SUPABASE = "supabase"
    LANGFUSE = "langfuse"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ProviderModel(StrictModel):
    provider: str
    model: str


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Usage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class TraceRequest(StrictModel):
    provider: str
    model: str
    system: str | None = None
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(StrictModel):
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    served_model: str | None = None


class TraceMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    tags: list[str] = Field(default_factory=list)
    pii_scrubbed: bool = False
    template_id: str | None = None


class TraceRecord(StrictModel):
    trace_id: UUID
    ts: datetime
    source: TraceSource
    workload: str
    request: TraceRequest
    response: TraceResponse = Field(default_factory=TraceResponse)
    meta: TraceMeta = Field(default_factory=TraceMeta)

    @model_validator(mode="after")
    def validate_single_turn(self) -> TraceRecord:
        if len(self.request.messages) != 1 or self.request.messages[0].role != "user":
            raise ValueError("TraceRecord must contain exactly one user message.")
        has_response_content = bool(self.response.text) or bool(self.response.tool_calls)
        if self.source != TraceSource.BACKFILL and not has_response_content:
            raise ValueError("TraceRecord response may be empty only when source is backfill.")
        return self


class SamplingConfig(StrictModel):
    n: int = Field(gt=0)
    stratify_by: list[str] = Field(default_factory=list)
    seed: int = 42
    k_baseline: int = Field(default=2, ge=2)


class AssertionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal[
        "contains",
        "not_contains",
        "regex",
        "json_schema",
        "json_valid",
        "numeric_field_equal",
        "length_range",
        "latency_p95_ms",
        "refusal",
        "judge_classifier",
    ]
    severity: Severity
    schema_ref: str | None = None
    prompt_ref: str | None = None
    min_chars: int | None = Field(default=None, ge=0)
    max_chars: int | None = Field(default=None, ge=0)
    max: float | None = None


class JudgeModelConfig(StrictModel):
    model: str
    # SPEC-NOTE: the pinned judge-entry shape is {provider, model} with a bare
    # model id. `provider` is optional so legacy "{provider}/{model}" strings
    # (family derived from the prefix) keep validating unchanged.
    provider: str | None = None


class JudgingConfig(StrictModel):
    families_allowed: list[str]
    min_judges: int = Field(ge=1)
    judges: list[JudgeModelConfig] | None = None


class ThresholdConfig(StrictModel):
    epsilon_pp: float = Field(ge=0)
    max_critical: int = Field(ge=0)
    confidence: float = Field(gt=0, lt=1)


class PrivacyConfig(StrictModel):
    scrub: bool = True
    egress: Literal["local", "in_tenant", "hosted_api"]


class BudgetConfig(StrictModel):
    max_usd: float = Field(ge=0)
    use_batch_api: bool = False


class ClusteringConfig(StrictModel):
    min_cluster_size: int | None = Field(default=None, ge=2)


class PatchConfig(StrictModel):
    min_fix_rate: float = Field(default=0.8, ge=0, le=1)


class MonitorConfig(StrictModel):
    suite_size: int = Field(default=150, gt=0)
    epsilon_pp: float = Field(default=0.0, ge=0)


class WorkloadManifest(StrictModel):
    workload: str
    baseline: ProviderModel
    candidate: ProviderModel
    sampling: SamplingConfig
    assertions: list[AssertionConfig]
    judging: JudgingConfig
    thresholds: ThresholdConfig
    privacy: PrivacyConfig
    budget: BudgetConfig
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)


class ContentHash(StrictModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str


class ParamTransformation(StrictModel):
    provider: str
    model: str
    path: str
    action: Literal["strip", "rename", "clamp"]
    original: Any | None = None
    transformed: Any | None = None
    reason: str
    count: int = Field(default=1, ge=1)


class CostLedgerEntry(StrictModel):
    stage: str
    provider: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    usd: float = Field(ge=0)


class RunManifest(StrictModel):
    run_id: str
    manifest_hash: ContentHash
    sampled_trace_hash: ContentHash
    engine_version: str
    seeds: dict[str, int]
    endpoints: dict[str, str]
    served_models: list[str]
    degraded: bool = False
    param_transformations: list[ParamTransformation] = Field(default_factory=list)
    wall_clock_seconds: float = Field(ge=0)
    cost_ledger: list[CostLedgerEntry] = Field(default_factory=list)
    run_started_at: str | None = None
    stages: dict[str, Any] = Field(default_factory=dict)


class Coverage(StrictModel):
    n: int = Field(gt=0)
    sampling_method: str
    strata: dict[str, int] = Field(default_factory=dict)
    zero_event_bound: str
    exclusion_percent: float = Field(ge=0, le=100)


class ConfidenceInterval(StrictModel):
    mean: float
    lower: float
    upper: float
    confidence: float


class AssertionResult(StrictModel):
    assertion_id: str
    severity: Severity
    failures: int = Field(ge=0)
    rate: float = Field(ge=0, le=1)


class JudgePanelSummary(StrictModel):
    models: list[str]
    families: list[str]
    order_swap: bool
    krippendorff_alpha: float | None = None
    agreement_percent: float | None = Field(default=None, ge=0, le=100)
    human_calibration_accuracy: float | None = Field(default=None, ge=0, le=1)


class RegressionCluster(StrictModel):
    name: str
    count: int = Field(ge=0)
    redacted_exemplars: list[str] = Field(default_factory=list)
    patch_diff: str | None = None
    patch_status: str | None = None


class Certificate(StrictModel):
    verdict: Literal["PASS", "BLOCK"]
    scope: str
    coverage: Coverage
    noise_floor: ConfidenceInterval | None
    candidate_disagreement: ConfidenceInterval
    delta: ConfidenceInterval
    assertion_results: list[AssertionResult]
    judge_panel: JudgePanelSummary | None = None
    regression_clusters: list[RegressionCluster] = Field(default_factory=list)
    cost_delta: dict[str, Any] = Field(default_factory=dict)
    latency_delta: dict[str, Any] = Field(default_factory=dict)
    config_hash: ContentHash
    engine_version: str
    public_key: str
    signature: str


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "trace-record": TraceRecord,
    "workload-manifest": WorkloadManifest,
    "run-manifest": RunManifest,
    "certificate": Certificate,
}


def export_json_schemas(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        path = output_dir / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written
