"""Experiment metadata: RunSpec for train/eval identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1


@dataclass
class RunSpec:
    task: str
    dynamics: str
    algorithm: str
    network: str = "mlp"
    sensor: str = "state"
    seed: int | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    algorithm_config: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    git_commit: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {key: value for key, value in data.items() if key in known}
        return cls(**payload)

    @classmethod
    def stamp(cls, **kwargs) -> "RunSpec":
        kwargs.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        return cls(**kwargs)
