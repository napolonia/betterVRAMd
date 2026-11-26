
from enum import Enum
from typing import Any, Protocol

from vramd.core.config import VRAMdConfig, load_config
from vramd.core.budget import ResourceBudgetCalculator
from vramd.core.lifecycle import LifecycleController
from vramd.core.scheduler import InferenceScheduler

from vramd.models.registry import ModelRegistry
from vramd.models.cache import RAMCacheManager

from vramd.monitor.system import SystemMonitor
from vramd.monitor.gpu import GPUMonitor


class ManagerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class ManagedComponent(Protocol):
    """Interface expected from manager-owned components."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class VRAMdManager:
    """Central coordinator for betterVRAMd."""

    def __init__(self, config: VRAMdConfig | None = None):
        self.config = config or load_config()
        self.state = ManagerState.STOPPED

        self.components: dict[str, ManagedComponent] = {}
        self._started_components: list[str] = []

        self.last_error: str | None = None

        # Model management: owned by VRAMdManager.
        self.model_registry = ModelRegistry(
            self.config.model_directory
        )

        self.ram_cache = RAMCacheManager()

        self.lifecycle = LifecycleController(
            registry=self.model_registry,
            cache=self.ram_cache,
        )

        # Inference scheduling.
        self.scheduler = InferenceScheduler(self.config)

        # Resource monitoring.
        self.system_monitor = SystemMonitor()

        self.gpu_monitor = (
            GPUMonitor(self.config.device_index)
            if self.config.gpu_enabled
            else None
        )

        self.register_component(
            "system_monitor",
            self.system_monitor,
        )

        if self.gpu_monitor is not None:
            self.register_component(
                "gpu_monitor",
                self.gpu_monitor,
            )

        self.budget_calculator = ResourceBudgetCalculator(
            self.config
        )

    def register_component(
        self,
        name: str,
        component: ManagedComponent,
    ) -> None:
        """Register a component before manager startup."""

        if self.state != ManagerState.STOPPED:
            raise RuntimeError(
                "Components can only be registered while stopped."
            )

        if name in self.components:
            raise ValueError(f"Component already registered: {name}")

        self.components[name] = component

    def start(self) -> None:
        """Start registered application components."""

        if self.state == ManagerState.RUNNING:
            return

        if self.state != ManagerState.STOPPED:
            raise RuntimeError(
                f"Cannot start manager in state: {self.state.value}"
            )

        self.state = ManagerState.STARTING
        self.last_error = None

        try:
            for name, component in self.components.items():
                self._started_components.append(name)
                component.start()

            self.state = ManagerState.RUNNING

        except Exception as error:
            self.last_error = str(error)
            self.state = ManagerState.FAILED

            errors = self._stop_started_components()

            try:
                self.lifecycle.shutdown()
            except Exception as cleanup_error:
                errors.append(f"lifecycle: {cleanup_error}")

            if errors:
                self.last_error += "; cleanup: " + "; ".join(errors)

            raise RuntimeError(
                f"betterVRAMd startup failed: {self.last_error}"
            ) from error

    def stop(self) -> None:
        """Stop components and release cached model mappings."""

        if self.state not in (
            ManagerState.STOPPED,
            ManagerState.FAILED,
        ):
            self.state = ManagerState.STOPPING

        errors = self._stop_started_components()

        try:
            self.lifecycle.shutdown()
        except Exception as error:
            errors.append(f"lifecycle: {error}")

        if errors:
            self.state = ManagerState.FAILED
            self.last_error = "; ".join(errors)

            raise RuntimeError(
                f"Shutdown encountered errors: {self.last_error}"
            )

        self.state = ManagerState.STOPPED
        self.last_error = None

    def _stop_started_components(self) -> list[str]:
        """Stop components in reverse startup order."""

        errors = []

        for name in reversed(self._started_components):
            component = self.components[name]

            try:
                component.stop()
            except Exception as error:
                errors.append(f"{name}: {error}")

        self._started_components.clear()
        return errors

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of manager state."""

        return {
            "application": self.config.app_name,
            "version": self.config.version,
            "state": self.state.value,
            "registered_components": list(self.components.keys()),
            "active_components": list(self._started_components),
            "registered_models": len(self.model_registry.models),
            "cached_models": len(self.ram_cache.list_cached()),
            "scheduler": self.scheduler.get_status(),
            "last_error": self.last_error,
        }

    def get_resource_status(self) -> dict:
        """Collect passive system and GPU measurements."""

        status = {
            "system": self.system_monitor.get_status(),
            "gpu": None,
        }

        if self.gpu_monitor is not None:
            status["gpu"] = self.gpu_monitor.get_status()

        return status

    def get_resource_budget(self):
        """Calculate safe resource limits from current measurements."""

        status = self.get_resource_status()

        return self.budget_calculator.calculate(
            system=status["system"],
            gpu=status["gpu"],
        )
