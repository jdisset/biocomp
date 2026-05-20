# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from biocomp.step_history import StepHistoryLike
from typing import Protocol, runtime_checkable


@runtime_checkable
class LoggerDispatch(Protocol):
    def on_start(self, config: object, stack: object) -> None: ...
    def on_step(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None: ...
    def on_end(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None: ...
    def needs_params_sync(self, step: int) -> bool: ...


class NullDispatch:
    __slots__ = ()

    def on_start(self, config: object, stack: object) -> None:
        pass

    def on_step(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        pass

    def on_end(
        self, step: int, config: object, step_history: StepHistoryLike, stack: object
    ) -> None:
        pass

    def needs_params_sync(self, step: int) -> bool:
        return False
