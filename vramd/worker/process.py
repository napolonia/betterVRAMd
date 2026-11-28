
"""Manage the llama-server subprocess lifecycle."""

import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from vramd.core.config import VRAMdConfig
from vramd.inference.llama_cpp import LlamaCppBackend

logger = logging.getLogger(__name__)


class WorkerProcessError(RuntimeError):
    """Base exception for worker process errors."""


class WorkerStartupError(WorkerProcessError):
    """Raised when the inference server fails to start."""


class WorkerShutdownError(WorkerProcessError):
    """Raised when the inference server cannot be stopped."""


class WorkerProcess:
    """Own the llama-server subprocess lifecycle."""

    def __init__(self, config: VRAMdConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._backend: LlamaCppBackend | None = None
        self._port: int | None = None

    @property
    def process(self) -> subprocess.Popen | None:
        """Return the managed subprocess."""
        return self._process

    @property
    def backend(self) -> LlamaCppBackend | None:
        """Return the backend associated with the server."""
        return self._backend

    @property
    def port(self) -> int | None:
        """Return the selected server port."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Check whether the subprocess is running."""
        return (
            self._process is not None
            and self._process.poll() is None
        )

    @staticmethod
    def _find_available_port(host: str) -> int:
        """Find an available port.

        The socket is released before llama-server binds it,
        so this method cannot guarantee exclusive allocation.
        """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]

    def start(
        self,
        model_path: str | Path,
        gpu_layers: int,
        mmproj_path: str | Path | None = None,
    ) -> LlamaCppBackend:
        """Start llama-server and wait until it becomes ready."""

        if self.is_running:
            raise WorkerProcessError(
                "An inference worker is already running."
            )

        # A previous process may have exited without being cleared.
        if self._process is not None:
            self.stop()

        self._clear_state()

        try:
            self._port = (
                self._find_available_port(self.config.server_host)
                if self.config.server_dynamic_port #TODO: allocation race, not atomic
                else self.config.server_port
            )

            backend = LlamaCppBackend(
                self.config,
                port=self._port,
            )

            command = backend.build_command(
                model_path=model_path,
                gpu_layers=gpu_layers,
                port=self._port,
                mmproj_path=mmproj_path,
            )

            logger.info(
                "Starting llama-server on %s:%s",
                self.config.server_host,
                self._port,
            )

            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )

            self._backend = backend

            self._wait_until_ready()

            logger.info(
                "llama-server ready (PID %s, port %s)",
                self._process.pid,
                self._port,
            )

            return backend

        except Exception as exc:
            try:
                self._cleanup_failed_start()

            except WorkerShutdownError as cleanup_error:
                raise WorkerStartupError(
                    "llama-server startup failed and process cleanup "
                    f"also failed: {cleanup_error}"
                ) from exc

            if isinstance(exc, WorkerProcessError):
                raise

            raise WorkerStartupError(
                f"Failed to start llama-server: {exc}"
            ) from exc

    def _wait_until_ready(self) -> None:
        """Wait for readiness while monitoring the subprocess."""

        deadline = (
            time.monotonic()
            + self.config.server_startup_timeout_seconds
        )

        while time.monotonic() < deadline:

            if self._process is None:
                raise WorkerStartupError(
                    "Worker process disappeared during startup."
                )

            return_code = self._process.poll()

            if return_code is not None:
                raise WorkerStartupError(
                    "llama-server exited during startup "
                    f"(exit code {return_code})."
                )

            if (
                self._backend is not None
                and self._backend.is_ready()
            ):
                return

            time.sleep(
                self.config.server_health_check_interval_seconds
            )

        raise WorkerStartupError(
            "llama-server startup timed out."
        )

    @staticmethod
    def _send_signal(
        process: subprocess.Popen,
        sig: signal.Signals,
    ) -> None:
        """Send a signal to the managed process group."""

        try:
            os.killpg(process.pid, sig)

        except ProcessLookupError:
            pass

        except OSError:
            if process.poll() is None:
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

    @staticmethod
    def _process_group_exists(pgid: int) -> bool:
        """Return whether a process group still exists."""

        try:
            os.killpg(pgid, 0)
            return True

        except ProcessLookupError:
            return False

        except PermissionError:
            return True


    @classmethod
    def _wait_for_process_group_exit(
        cls,
        process: subprocess.Popen,
        timeout: float,
    ) -> bool:
        """Wait until the process group and its direct child exit."""

        deadline = time.monotonic() + timeout

        while True:
            # Reap the direct child when it has exited.
            process.poll()

            if not cls._process_group_exists(process.pid):
                return True

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return False

            time.sleep(min(0.05, remaining))



    def stop(self) -> None:
        """Stop the server and verify its entire process group exits."""

        process = self._process

        if process is None:
            self._clear_state()
            return

        logger.info(
            "Stopping llama-server (PID %s)",
            process.pid,
        )

        try:
            self._send_signal(process, signal.SIGTERM)

            stopped = self._wait_for_process_group_exit(
                process,
                self.config.shutdown_timeout_seconds,
            )

            if not stopped:
                if not self.config.force_termination:
                    raise WorkerShutdownError(
                        "llama-server process group did not stop "
                        "within the configured timeout. "
                        "Process remains tracked."
                    )

                logger.warning(
                    "Graceful shutdown timed out; "
                    "forcing process-group termination."
                )

                self._send_signal(process, signal.SIGKILL)

                stopped = self._wait_for_process_group_exit(
                    process,
                    self.config.shutdown_timeout_seconds,
                )

                if not stopped:
                    raise WorkerShutdownError(
                        "Unable to terminate llama-server process group. "
                        "Process remains tracked."
                    )

            # Explicitly confirm collection of the direct child.
            process.wait(timeout=0)

        except WorkerShutdownError:
            raise

        except (subprocess.TimeoutExpired, OSError) as exc:
            raise WorkerShutdownError(
                "Failed to completely stop llama-server. "
                "Process remains tracked."
            ) from exc

        logger.info("llama-server process group stopped.")

        self._clear_state()



    def _cleanup_failed_start(self) -> None:
        """Clean up the entire process group after failed startup."""

        process = self._process

        if process is None:
            self._clear_state()
            return

        logger.warning(
            "Cleaning up failed llama-server startup (PID %s)",
            process.pid,
        )

        try:
            self._send_signal(process, signal.SIGKILL)

            stopped = self._wait_for_process_group_exit(
                process,
                self.config.shutdown_timeout_seconds,
            )

            if not stopped:
                raise WorkerShutdownError(
                    "Failed to terminate llama-server process group. "
                    "Process remains tracked."
                )

            process.wait(timeout=0)

        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.exception(
                "Failed to clean up llama-server process %s.",
                process.pid,
            )

            raise WorkerShutdownError(
                "Failed to clean up llama-server. "
                "Process remains tracked."
            ) from exc

        self._clear_state()


    def _clear_state(self) -> None:
        """Clear references after the process has stopped."""

        self._process = None
        self._backend = None
        self._port = None
