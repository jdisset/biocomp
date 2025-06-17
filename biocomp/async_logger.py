import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable, Any, Tuple
import traceback

logger = logging.getLogger(__name__)


class AsyncLoggerManager:
    """Manages asynchronous execution of logger callbacks during training"""

    def __init__(self, max_workers: int = 4, sync_loggers: bool = False):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="logger")
        self.pending_tasks: List[asyncio.Task] = []
        self._shutdown = False
        self.sync_loggers = sync_loggers

    def _should_call(self, step: int, period: Optional[int]) -> bool:
        """Determine if a logger callback should be called at this step"""
        if period is None or period == -1:
            return False  # end-of-training loggers
        if period == 0:
            return step == 0  # start-of-training loggers
        return step % period == 0

    async def _run_callback_safe(self, callback: Callable, step: int, **kwargs) -> None:
        """Run a single callback with error handling"""
        try:
            if self.sync_loggers:
                callback(
                    step,
                    kwargs.get("training_config"),
                    kwargs.get("step_history"),
                    kwargs.get("stack"),
                )
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    callback,
                    step,
                    kwargs.get("training_config"),
                    kwargs.get("step_history"),
                    kwargs.get("stack"),
                )
        except Exception as e:
            logger.error(f"Logger callback failed at step {step}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

    async def submit_logger_batch(
        self, step: int, logger_callbacks: List[Tuple[int, Callable]], **kwargs
    ) -> None:
        """Submit all applicable logger callbacks for a step as async tasks"""
        if self._shutdown:
            return

        # wait for previous step's loggers to complete before starting new ones
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            self.pending_tasks.clear()

        if self.sync_loggers:
            # run loggers synchronously, one after another
            for period, callback in logger_callbacks:
                if self._should_call(step, period):
                    # create lightweight copy of step_history for the logger
                    step_history_copy = self._create_copy(kwargs.get("step_history"))
                    kwargs_copy = {**kwargs, "step_history": step_history_copy}

                    await self._run_callback_safe(callback, step, **kwargs_copy)

            logger.debug(
                f"Executed {sum(1 for p, _ in logger_callbacks if self._should_call(step, p))} loggers synchronously for step {step}"
            )
        else:
            # create new tasks for this step (async mode)
            new_tasks = []
            for period, callback in logger_callbacks:
                if self._should_call(step, period):
                    # create lightweight copy of step_history for the logger
                    step_history_copy = self._create_copy(kwargs.get("step_history"))
                    kwargs_copy = {**kwargs, "step_history": step_history_copy}

                    task = asyncio.create_task(
                        self._run_callback_safe(callback, step, **kwargs_copy)
                    )
                    new_tasks.append(task)

            # set new tasks as pending (replace, don't accumulate)
            self.pending_tasks = new_tasks
            logger.debug(f"Submitted {len(new_tasks)} logger tasks for step {step}")

    async def wait_for_all_loggers(self) -> None:
        """Wait for all pending logger tasks to complete"""
        if self.pending_tasks:
            try:
                await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error waiting for logger tasks: {e}")
            finally:
                self.pending_tasks.clear()

    def _create_copy(self, step_history):
        """Create a lightweight copy of step_history for async processing"""
        if step_history is None:
            return None

        import copy
        import jax
        import numpy as np
        from biocomp.jaxutils import tree_to_np
        from biocomp.parameters import ParameterTree, PTree

        # create deep copy for all data to avoid shared references
        step_history_copy = {}
        for key, value in step_history.items():
            if isinstance(value, (ParameterTree, PTree)):
                step_history_copy[key] = tree_to_np(value)
            elif isinstance(value, (list, np.ndarray)):
                step_history_copy[key] = copy.deepcopy(value)
            else:
                step_history_copy[key] = value

        return step_history_copy

    async def submit_end_loggers(
        self, step: int, logger_callbacks: List[Tuple[int, Callable]], **kwargs
    ) -> None:
        """Submit and wait for end-of-training loggers (period=None or -1)"""
        end_tasks = []
        for period, callback in logger_callbacks:
            if period is None or period == -1:
                task = asyncio.create_task(self._run_callback_safe(callback, step, **kwargs))
                end_tasks.append(task)

        if end_tasks:
            await asyncio.gather(*end_tasks, return_exceptions=True)

    def shutdown(self):
        """Shutdown the thread pool executor"""
        self._shutdown = True
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.wait_for_all_loggers()
        self.shutdown()
