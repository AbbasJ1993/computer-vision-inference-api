"""Thread-owned model pool with bounded admission and queue timeouts.

Each worker owns a separate InferenceService/model and executes only on its
own single-thread executor. ``MAX_QUEUE_SIZE`` counts admitted requests waiting
for a free model, in addition to the active workers. Admission is non-blocking.
A request that has already started inference retains its capacity slot until
the underlying thread actually finishes, even if its HTTP caller is cancelled.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, Sequence

from app.services.inference_service import InferenceProfile, InferenceResult, InferenceService


class InferenceOverloadedError(Exception):
    """No capacity to accept a new request (HTTP 503)."""


class InferenceQueueTimeoutError(Exception):
    """Admitted request waited too long for an idle model (HTTP 503)."""


@dataclass(frozen=True)
class PoolInferenceResult:
    result: InferenceResult
    profiling: InferenceProfile | None
    queue_wait_ms: float
    inference_time_ms: float
    model_worker_id: int


class InferenceAdmission:
    """One capacity reservation, safe to release before inference starts.

    The pool owns release once a model task has been submitted; cancelling the
    HTTP request must NOT release the slot while its inference is still running.
    """

    def __init__(self, pool: InferenceWorkerPool) -> None:
        self._pool = pool
        self._active = True
        self._claimed = False
        self._running = False

    def release_if_pending(self) -> None:
        """Return capacity when upload/queue acquisition fails or is cancelled."""
        if self._active and not self._running:
            self._active = False
            self._pool._admitted -= 1

    def _complete(self) -> None:
        """Called only after the underlying worker future actually completes."""
        if self._active:
            self._active = False
            self._pool._admitted -= 1


def _infer_on_owned_thread(
    service: InferenceService,
    image_bytes: bytes,
    confidence_threshold: float,
    profile: bool,
) -> tuple[InferenceResult, InferenceProfile | None]:
    if profile:
        return service.infer_profiled(image_bytes, confidence_threshold)
    return service.infer(image_bytes, confidence_threshold), None


class InferenceWorkerPool:
    """An independent model on each dedicated thread, with bounded admission.

    At most ``worker_count + max_queue_size`` requests may be admitted. An
    admitted request can wait at most ``queue_timeout_seconds`` for a model.
    There is no unbounded backlog in ThreadPoolExecutor.

    Use ``reserve()`` before reading an upload and pass the ticket to ``infer``;
    direct callers may omit it, in which case ``infer`` reserves automatically.
    These methods are intended to run on the ASGI application's event loop.
    """

    def __init__(
        self,
        factories: Sequence[Callable[[], InferenceService]],
        *,
        max_queue_size: int = 16,
        queue_timeout_seconds: float = 2.0,
    ) -> None:
        if not 1 <= len(factories) <= 2:
            raise ValueError("This experiment supports one or two model workers.")
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 0:
            raise ValueError("max_queue_size must be a non-negative integer.")
        if not isfinite(queue_timeout_seconds) or queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be finite and positive.")
        self._factories = tuple(factories)
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self._executors: list[ThreadPoolExecutor] = []
        self._services: list[InferenceService] = []
        self._available: asyncio.Queue[int] = asyncio.Queue()
        self._started = False
        self._closed = False
        self._admitted = 0

    @property
    def worker_count(self) -> int:
        return len(self._factories)

    @property
    def admitted_count(self) -> int:
        """Active plus waiting reservations (including upload being read)."""
        return self._admitted

    @property
    def max_admitted(self) -> int:
        return self.worker_count + self.max_queue_size

    def reserve(self) -> InferenceAdmission:
        """Reject immediately if full; no suspension or unbounded waiting."""
        if not self._started or self._closed:
            raise RuntimeError("Inference worker pool is not running.")
        # On the single ASGI event loop, check+increment is atomic: no await.
        if self._admitted >= self.max_admitted:
            raise InferenceOverloadedError("Inference capacity exceeded; retry later.")
        self._admitted += 1
        return InferenceAdmission(self)

    async def start(self) -> None:
        """Load each service on its own permanent thread at ASGI startup."""
        if self._started or self._closed:
            raise RuntimeError("Pool already started or closed.")
        try:
            for index, factory in enumerate(self._factories):
                executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"yolo-owner-{index}"
                )
                self._executors.append(executor)
                service = await asyncio.wrap_future(executor.submit(factory))
                self._services.append(service)
            for index in range(self.worker_count):
                self._available.put_nowait(index)
            self._started = True
        except BaseException:
            await self.close()
            raise

    async def infer(
        self,
        image_bytes: bytes,
        confidence_threshold: float,
        *,
        profile: bool,
        admission: InferenceAdmission | None = None,
    ) -> PoolInferenceResult:
        """Wait a bounded time for a model, execute on its owner thread.

        Cancellation while *waiting* releases capacity. Cancellation during
        active inference does not prematurely free its model or capacity.
        """
        if not self._started or self._closed:
            if admission is not None:
                admission.release_if_pending()
            raise RuntimeError("Inference worker pool is not running.")
        ticket = admission if admission is not None else self.reserve()
        if ticket._pool is not self or not ticket._active or ticket._claimed:
            raise ValueError("Invalid or already-used admission ticket.")
        ticket._claimed = True
        worker_id: int | None = None
        try:
            queued_at = perf_counter()
            try:
                worker_id = await asyncio.wait_for(
                    self._available.get(), timeout=self.queue_timeout_seconds
                )
                if self._closed:
                    raise InferenceOverloadedError("Inference pool is shutting down.")
            except TimeoutError as exc:
                raise InferenceQueueTimeoutError(
                    "Timed out waiting for an inference worker."
                ) from exc

            queue_wait_ms = round((perf_counter() - queued_at) * 1000, 3)
            started = perf_counter()
            loop = asyncio.get_running_loop()
            future = self._executors[worker_id].submit(
                _infer_on_owned_thread,
                self._services[worker_id],
                image_bytes,
                confidence_threshold,
                profile,
            )
            # From here on, the *underlying thread* owns the capacity ticket.
            ticket._running = True

            assigned_worker_id = worker_id

            def _release_after_completion(_completed) -> None:
                loop.call_soon_threadsafe(self._finish, assigned_worker_id, ticket)

            future.add_done_callback(_release_after_completion)
            worker_id = None  # ownership moved to completion callback

            result, timings = await asyncio.shield(asyncio.wrap_future(future))
            inference_time_ms = round((perf_counter() - started) * 1000, 3)
            return PoolInferenceResult(
                result=result,
                profiling=timings,
                queue_wait_ms=queue_wait_ms,
                inference_time_ms=inference_time_ms,
                model_worker_id=assigned_worker_id,
            )
        finally:
            # If cancellation/exception occurred before submitting work, both
            # worker ID (if acquired) and admission capacity must be returned.
            if worker_id is not None:
                self._available.put_nowait(worker_id)
            ticket.release_if_pending()

    def _finish(self, worker_id: int, ticket: InferenceAdmission) -> None:
        # The worker becomes available only after its future finishes.
        self._available.put_nowait(worker_id)
        ticket._complete()

    async def close(self) -> None:
        """Wait for active inferences without blocking the event loop."""
        if self._closed:
            return
        self._closed = True
        for executor in tuple(self._executors):
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
        # Drain scheduled done callbacks before clearing bookkeeping.
        await asyncio.sleep(0)
        self._services.clear()
        self._executors.clear()
        self._started = False
