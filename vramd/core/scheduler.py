
"""Inference request scheduling and queue management."""

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from vramd.core.config import VRAMdConfig


class RequestStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    request_id: int
    model_name: str
    prompt: str
    priority: int = 0
    status: RequestStatus = RequestStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: Any = None


class InferenceScheduler:
    """Manage inference requests and scheduling order."""

    def __init__(self, config: VRAMdConfig):
        self._lock = threading.RLock() #to protect all shared scheduler state

        self.enabled = config.scheduler_enabled
        self.queue_enabled = config.queue_enabled
        self.max_queue_size = config.max_queue_size
        self.priority_enabled = config.priority_enabled
        self.history_limit = config.scheduler_history_limit

        self._request_ids = itertools.count()
        self._sequence = itertools.count()
        self._queue: list[tuple[int, int, int]] = []
        self._requests: dict[int, InferenceRequest] = {}


    def submit(
        self,
        model_name: str,
        prompt: str,
        priority: int = 0,
    ) -> InferenceRequest:
        """Register and enqueue an inference request."""
        with self._lock:
            if not self.enabled:
                raise RuntimeError("Scheduler is disabled.")

            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("Model name cannot be empty.")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt cannot be empty.")

            if not isinstance(priority, int) or isinstance(priority, bool):
                raise TypeError("Priority must be an integer.")

            queued_count = sum(
                request.status == RequestStatus.QUEUED
                for request in self._requests.values()
            )

            running_count = sum(
                request.status == RequestStatus.RUNNING
                for request in self._requests.values()
            )

            if not self.queue_enabled and (queued_count or running_count):
                raise RuntimeError(
                    "Request queueing is disabled and another request "
                    "is already active."
                )

            if queued_count >= self.max_queue_size:
                raise RuntimeError("Inference queue is full.")

            request_id = next(self._request_ids)

            request = InferenceRequest(
                request_id=request_id,
                model_name=model_name,
                prompt=prompt,
                priority=priority if self.priority_enabled else 0,
            )

            sequence = next(self._sequence)

            heapq.heappush(
                self._queue,
                (-request.priority, sequence, request_id),
            )

            self._requests[request_id] = request

            return request

    def next_request(self) -> InferenceRequest | None:
        """Retrieve the next queued request for execution."""
        with self._lock:

            # Enforce single-request execution.
            if any(
                request.status == RequestStatus.RUNNING
                for request in self._requests.values()
            ):
                return None

            while self._queue:
                _, _, request_id = heapq.heappop(self._queue)
                request = self._requests.get(request_id)

                if request is None:
                    continue

                if request.status != RequestStatus.QUEUED:
                    continue

                request.status = RequestStatus.RUNNING
                request.started_at = time.time()

                return request

            return None

    def complete(self, request_id: int, result: Any = None) -> None:
        """Mark a running request as completed."""
        with self._lock:
            request = self.get_request(request_id)
            self._require_running(request)

            request.result = result
            request.status = RequestStatus.COMPLETED
            request.finished_at = time.time()

            self._prune_history()

    def fail(self, request_id: int, error: str) -> None:
        """Mark a running request as failed."""
        with self._lock:
            request = self.get_request(request_id)
            self._require_running(request)

            request.error = error
            request.status = RequestStatus.FAILED
            request.finished_at = time.time()
            
            self._prune_history()

    def cancel(self, request_id: int) -> None:
        """Cancel a queued request."""
        with self._lock:
            request = self.get_request(request_id)

            if request.status != RequestStatus.QUEUED:
                raise RuntimeError(
                    "Only queued requests can be cancelled."
                )

            request.status = RequestStatus.CANCELLED
            request.finished_at = time.time()
            
            self._prune_history()

    def get_request(self, request_id: int) -> InferenceRequest:
        """Retrieve a snapshot of a request by its ID."""
        with self._lock:
            if request_id not in self._requests:
                raise KeyError(f"Unknown request: {request_id}")

            # Return a shallow copy to prevent external modification of
            # internal scheduler state. Mutable fields such as result are
            # still shared references and must not be modified by callers.
            return replace(self._requests[request_id])

    def list_requests(self) -> list[InferenceRequest]:
        """Return snapshots of all registered requests."""
        with self._lock:
            # Shallow copies protect internal request state from external
            # modification. Mutable fields such as result remain shared.
            return [
                replace(request)
                for request in self._requests.values()
            ]

    def get_status(self) -> dict[str, bool | int]:
        """Return current scheduler statistics."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "queued": sum(
                    r.status == RequestStatus.QUEUED
                    for r in self._requests.values()
                ),
                "running": sum(
                    r.status == RequestStatus.RUNNING
                    for r in self._requests.values()
                ),
                "completed": sum(
                    r.status == RequestStatus.COMPLETED
                    for r in self._requests.values()
                ),
                "failed": sum(
                    r.status == RequestStatus.FAILED
                    for r in self._requests.values()
                ),
                "cancelled": sum(
                    r.status == RequestStatus.CANCELLED
                    for r in self._requests.values()
                ),
            }

    def _get_request_unlocked(
        self,
        request_id: int,
    ) -> InferenceRequest:
        """Retrieve an internal request. Caller must hold _lock."""

        if request_id not in self._requests:
            raise KeyError(f"Unknown request: {request_id}")

        return self._requests[request_id]

    def _prune_history(self) -> None:
        """Remove the oldest terminal requests beyond the history limit."""

        terminal_statuses = {
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
        }

        terminal_requests = sorted(
            (
                request
                for request in self._requests.values()
                if request.status in terminal_statuses
            ),
            key=lambda request: (
                request.finished_at,
                request.request_id,
            ),
        )

        excess = len(terminal_requests) - self.history_limit

        if excess <= 0:
            return

        for request in terminal_requests[:excess]:
            del self._requests[request.request_id]


    @staticmethod
    def _require_running(request: InferenceRequest) -> None:
        """Ensure a request is currently running."""

        if request.status != RequestStatus.RUNNING:
            raise RuntimeError(
                f"Request {request.request_id} is not running."
            )
