
"""Persistent llama.cpp server backend.

This module builds llama-server commands and communicates with
an already-running server.

Process creation and termination are owned by worker/process.py.
"""

import json
import os
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from collections.abc import Sequence

from vramd.inference.base import (
    BackendUnavailableError,
    InferenceBackend,
    InferenceBackendError,
    InferenceRequestError,
    InferenceResult,
    InferenceTimeoutError,
)


class LlamaCppBackend(InferenceBackend):
    """HTTP client and command builder for llama-server."""

    def __init__(self, config, port: int | None = None):
        self.config = config

        if config.server_dynamic_port and port is None:
            raise ValueError(
                "A dynamically selected server port must be provided by the worker."
            )

        self.host = config.server_host
        self.port = port if port is not None else config.server_port
        self.base_url = f"http://{self.host}:{self.port}"

        self.api_key_file = (
            Path(config.server_api_key_file).expanduser()
            if config.server_api_key_file is not None
            else None
        )
        self.api_key = None

        if self.api_key_file is not None:
            key_path = self.api_key_file

            if not key_path.is_file() or not os.access(key_path, os.R_OK):
                raise InferenceBackendError(
                    "Configured server API key file is missing or unreadable."
                )

            self.api_key = key_path.read_text(encoding="utf-8").strip()

            if not self.api_key:
                raise InferenceBackendError("Configured server API key file is empty.")

        self.request_timeout = config.server_request_timeout_seconds

    def validate_executable(self) -> Path:
        """Validate the configured llama-server executable."""

        executable = Path(self.config.llama_server).expanduser()

        if not executable.is_file():
            raise InferenceBackendError(
                f"llama-server executable not found: {executable}"
            )

        if not os.access(executable, os.X_OK):
            raise InferenceBackendError(f"llama-server is not executable: {executable}")

        return executable

    @staticmethod
    def validate_model(model_path: str | Path) -> Path:
        """Validate that the selected model file exists."""

        model = Path(model_path).expanduser()

        if not model.is_file():
            raise InferenceBackendError(f"Model file not found: {model}")

        return model.resolve()

    def build_command(
        self,
        model_path: str | Path,
        gpu_layers: int,
        port: int | None = None,
        mmproj_path: str | Path | None = None,
    ) -> list[str]:
        """Build the llama-server command without executing it."""

        executable = self.validate_executable()
        model = self.validate_model(model_path)

        if gpu_layers < 0:
            raise ValueError("gpu_layers cannot be negative")

        selected_port = port if port is not None else self.port

        if not 1 <= selected_port <= 65535:
            raise ValueError("Invalid server port")

        command = [
            str(executable),
            "--model",
            str(model),
            "--host",
            self.host,
            "--port",
            str(selected_port),
            "--ctx-size",
            str(self.config.context_size),
            "--n-gpu-layers",
            str(gpu_layers),
            "--threads",
            str(self.config.threads),
            "--parallel",
            str(self.config.server_max_parallel_requests),
            "--no-webui",
        ]

        if mmproj_path is not None:
            mmproj = self.validate_model(mmproj_path)
            command.extend(["--mmproj", str(mmproj)])

        if self.api_key_file is not None:
            command.extend(
                [
                    "--api-key-file",
                    str(self.api_key_file),
                ]
            )

        return command

    def _headers(self) -> dict[str, str]:
        """Build HTTP request headers."""

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def is_ready(self) -> bool:
        """Check whether llama-server is ready to accept requests."""

        request = Request(
            f"{self.base_url}/health",
            headers=self._headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=self.config.server_health_check_interval_seconds,
            ) as response:
                return response.status == 200

        except (HTTPError, URLError, OSError, TimeoutError):
            return False

    def wait_until_ready(self) -> None:
        """Wait for an already-started server to become ready."""

        deadline = time.monotonic() + self.config.server_startup_timeout_seconds

        while time.monotonic() < deadline:
            if self.is_ready():
                return

            time.sleep(self.config.server_health_check_interval_seconds)

        raise BackendUnavailableError(
            "llama-server did not become ready before the startup timeout."
        )

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
        """Submit an inference request to the persistent server."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")

        if not self.is_ready():
            raise BackendUnavailableError("llama-server is not ready.")

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")

        if stop is not None and not all(
            isinstance(item, str) and item for item in stop
        ):
            raise ValueError("stop must contain non-empty strings")

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        request = Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        request_timeout = timeout if timeout is not None else self.request_timeout

        try:
            with urlopen(request, timeout=request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            if exc.code in (502, 503, 504):
                raise BackendUnavailableError(
                    f"llama-server unavailable (HTTP {exc.code})."
                ) from exc

            raise InferenceRequestError(
                f"llama-server request failed (HTTP {exc.code})."
            ) from exc

        except (TimeoutError, socket.timeout) as exc:
            raise InferenceTimeoutError("Inference request timed out.") from exc

        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise InferenceTimeoutError("Inference request timed out.") from exc

            raise BackendUnavailableError(
                "Could not communicate with llama-server."
            ) from exc

        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceRequestError("Invalid response from llama-server.") from exc

        if not isinstance(data, dict) or not isinstance(data.get("content"), str):
            raise InferenceRequestError("llama-server returned an unexpected response.")

        timings = data.get("timings") or {}

        prompt_ms = timings.get("prompt_ms", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        duration = None

        if isinstance(prompt_ms, (int, float)) and isinstance(
            predicted_ms, (int, float)
        ):
            duration = (prompt_ms + predicted_ms) / 1000

        return InferenceResult(
            text=data["content"],
            prompt_tokens=data.get("tokens_evaluated"),
            completion_tokens=data.get("tokens_predicted"),
            total_duration_seconds=duration,
            finish_reason=data.get("stop_type"),
        )
