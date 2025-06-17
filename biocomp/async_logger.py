import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable, Any, Tuple
import traceback

logger = logging.getLogger(__name__)

class AsyncLoggerManager:
    """Manages asynchronous execution of logger callbacks during training"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="logger")
        self.pending_tasks: List[asyncio.Task] = []
        self._shutdown = False
    
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
            # run callback in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                callback,
                step,
                kwargs.get('training_config'),
                kwargs.get('step_history'),
                kwargs.get('xbatches'),
                kwargs.get('ybatches'),
                kwargs.get('stack')
            )
        except Exception as e:
            logger.error(f"Logger callback failed at step {step}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    async def submit_logger_batch(self, step: int, logger_callbacks: List[Tuple[int, Callable]], **kwargs) -> List[asyncio.Task]:
        """Submit all applicable logger callbacks for a step as async tasks"""
        if self._shutdown:
            return []
            
        tasks = []
        for period, callback in logger_callbacks:
            if self._should_call(step, period):
                task = asyncio.create_task(
                    self._run_callback_safe(callback, step, **kwargs)
                )
                tasks.append(task)
        
        return tasks
    
    async def wait_for_previous_loggers(self) -> None:
        """Wait for all pending logger tasks to complete"""
        if self.pending_tasks:
            try:
                await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error waiting for logger tasks: {e}")
            finally:
                self.pending_tasks.clear()
    
    async def submit_end_loggers(self, step: int, logger_callbacks: List[Tuple[int, Callable]], **kwargs) -> None:
        """Submit and wait for end-of-training loggers (period=None or -1)"""
        end_tasks = []
        for period, callback in logger_callbacks:
            if period is None or period == -1:
                task = asyncio.create_task(
                    self._run_callback_safe(callback, step, **kwargs)
                )
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
        await self.wait_for_previous_loggers()
        self.shutdown()