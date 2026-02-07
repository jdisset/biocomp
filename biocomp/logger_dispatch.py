from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LoggerDispatch(Protocol):
    def on_start(self, config: object, stack: object) -> None: ...
    def on_step(self, step: int, config: object, step_history: dict, stack: object) -> None: ...
    def on_end(self, step: int, config: object, step_history: dict, stack: object) -> None: ...
    def needs_params_sync(self, step: int) -> bool: ...


class NullDispatch:
    __slots__ = ()

    def on_start(self, config: object, stack: object) -> None:
        pass

    def on_step(self, step: int, config: object, step_history: dict, stack: object) -> None:
        pass

    def on_end(self, step: int, config: object, step_history: dict, stack: object) -> None:
        pass

    def needs_params_sync(self, step: int) -> bool:
        return False
