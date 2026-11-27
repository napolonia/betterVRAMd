
"""Common inference backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class InferenceBackendError(RuntimeError):
    """Base exception for inference backend errors."""


class BackendUnavailableError(InferenceBackendError):
    """Raised when the inference backend is unavailable."""


class InferenceRequestError(InferenceBackendError):
    """Raised when an inference request fails."""


class InferenceTimeoutError(InferenceBackendError):
    """Raised when an inference request exceeds its timeout."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Common result returned by inference backends."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_seconds: float | None = None
    finish_reason: str | None = None


class InferenceBackend(ABC):
    """
    Abstract interface for inference backends.

    Backend implementations handle communication with their
    respective inference engines.

    Process creation and termination belong to the worker.
    Resource budgeting and model lifecycle decisions belong
    to the core application.
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check whether the backend can accept inference requests.

        Must not start a backend or load a model.
        """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> InferenceResult:
        """
        Generate a response using the currently available model.

        Raises:
            BackendUnavailableError: Backend is not ready.
            InferenceRequestError: Request failed.
            InferenceTimeoutError: Request timed out.
        """
