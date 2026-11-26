
import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    def __init__(self, config):
        self.enabled = config.scheduler_enabled
        self.queue_enabled = config.queue_enabled
        self.max_queue_size = config.max_queue_size
        self.priority_enabled = config.priority_enabled

        self._sequence = itertools.count()
        self._queue = []
        self._requests: dict[int, InferenceRequest] = {}

    def submit(
        self,
        model_name: str,
        prompt: str,
        priority: int = 0,
    ) -> InferenceRequest:

        if not self.enabled:
            raise RuntimeError("Scheduler is disabled.")

        if not model_name:
            raise ValueError("Model name cannot be empty.")

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if len(self._queue) >= self.max_queue_size:
            raise RuntimeError("Inference queue is full.")

        if not self.queue_enabled and self._queue:
            raise RuntimeError("Request queueing is disabled.")

        request_id = next(self._sequence)

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
        """Retrieve the next request for execution."""

        while self._queue:
            _, _, request_id = heapq.heappop(self._queue)
            request = self._requests[request_id]

            if request.status != RequestStatus.QUEUED:
                continue

            request.status = RequestStatus.RUNNING
            request.started_at = time.time()

            return request

        return None

    def complete(self, request_id: int, result: Any = None) -> None:
        request = self.get_request(request_id)
        self._require_running(request)

        request.result = result
        request.status = RequestStatus.COMPLETED
        request.finished_at = time.time()

    def fail(self, request_id: int, error: str) -> None:
        request = self.get_request(request_id)
        self._require_running(request)

        request.error = error
        request.status = RequestStatus.FAILED
        request.finished_at = time.time()

    def cancel(self, request_id: int) -> None:
        request = self.get_request(request_id)

        if request.status != RequestStatus.QUEUED:
            raise RuntimeError("Only queued requests can be cancelled.")

        request.status = RequestStatus.CANCELLED
        request.finished_at = time.time()

    def get_request(self, request_id: int) -> InferenceRequest:
        if request_id not in self._requests:
            raise KeyError(f"Unknown request: {request_id}")

        return self._requests[request_id]

    def list_requests(self) -> list[InferenceRequest]:
        return list(self._requests.values())

    def get_status(self) -> dict:
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

    @staticmethod
    def _require_running(request: InferenceRequest) -> None:
        if request.status != RequestStatus.RUNNING:
            raise RuntimeError(
                f"Request {request.request_id} is not running."
            )
