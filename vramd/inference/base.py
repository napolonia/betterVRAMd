
"""Common inference backend interface."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence


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

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Result text must be a string.")

        for field_name in (
            "prompt_tokens",
            "completion_tokens",
        ):
            value = getattr(self, field_name)

            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{field_name} must be an integer or None."
                    )

                if value < 0:
                    raise ValueError(
                        f"{field_name} cannot be negative."
                    )

        duration = self.total_duration_seconds

        if duration is not None:
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
            ):
                raise TypeError(
                    "total_duration_seconds must be numeric or None."
                )

            if not math.isfinite(duration) or duration < 0:
                raise ValueError(
                    "total_duration_seconds must be finite "
                    "and non-negative."
                )

        if (
            self.finish_reason is not None
            and not isinstance(self.finish_reason, str)
        ):
            raise TypeError(
                "finish_reason must be a string or None."
            )


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
