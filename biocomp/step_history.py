from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, TypeAlias

StepHistoryLike: TypeAlias = Mapping[str, Any] | "StepHistorySnapshot"


@dataclass(frozen=True)
class StepHistorySnapshot(Mapping[str, Any]):
    """Typed wrapper around a finalized step-history mapping."""

    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: object) -> StepHistorySnapshot:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "Step history invariant violated: mapping keys must be strings, "
                        f"got key type {type(key).__name__}."
                    )
                data[key] = item
            return cls(data=data)
        raise TypeError(
            "Step history invariant violated: expected Mapping[str, Any] "
            f"or StepHistorySnapshot, got {type(value).__name__}."
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def ensure_step_history_snapshot(
    value: object,
    *,
    context: str = "step_history",
) -> StepHistorySnapshot:
    try:
        return StepHistorySnapshot.from_raw(value)
    except TypeError as exc:
        raise TypeError(
            f"{context} invariant violated: expected Mapping[str, Any] "
            f"or StepHistorySnapshot, got {type(value).__name__}."
        ) from exc
