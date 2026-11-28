
"""Central application manager for betterVRAMd."""

from enum import Enum
from typing import Any, Protocol

from vramd.core.budget import ResourceBudgetCalculator
from vramd.core.config import VRAMdConfig, load_config
from vramd.core.lifecycle import LifecycleController
from vramd.core.scheduler import InferenceScheduler
from vramd.models.cache import RAMCacheManager
from vramd.models.estimator import ModelMemoryEstimator
from vramd.models.metadata import ModelMetadataInspector
from vramd.models.registry import ModelRegistry
from vramd.models.selector import ModelLayerSelector
from vramd.monitor.gpu import GPUMonitor
from vramd.monitor.system import SystemMonitor

#ARCHITECT detail: the manager currently does not own a dedicated inference execution loop. 
#   The scheduler can accept and track requests, but it cannot execute them yet. We'll address that separately.
# Option A: Use llama.cpp's built-in HTTP server
# Option B: Build our own inference interface
# Using A, B for future work.

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

        # Model management.
        self.model_registry = ModelRegistry(
            self.config.model_directory
        )

        self.ram_cache = RAMCacheManager()

        self.metadata_inspector = ModelMetadataInspector(
            self.config
        )

        # Memory estimation and GPU-layer selection.
        self.memory_estimator = ModelMemoryEstimator()

        self.layer_selector = ModelLayerSelector(
            estimator=self.memory_estimator
        )

        # Lifecycle coordination.
        self.lifecycle = LifecycleController(
            registry=self.model_registry,
            cache=self.ram_cache,
            metadata_inspector=self.metadata_inspector,
            layer_selector=self.layer_selector,
            config=self.config,
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

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Component name cannot be empty."
            )

        if name in self.components:
            raise ValueError(
                f"Component already registered: {name}"
            )

        if not callable(getattr(component, "start", None)):
            raise TypeError(
                "Managed component must implement start()."
            )

        if not callable(getattr(component, "stop", None)):
            raise TypeError(
                "Managed component must implement stop()."
            )

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
                # Record before starting so a partially initialized
                # component also receives a cleanup attempt.
                self._started_components.append(name)
                component.start()

            self.state = ManagerState.RUNNING

        except Exception as error:
            self.last_error = str(error)
            self.state = ManagerState.FAILED

            cleanup_errors = self._stop_started_components()

            try:
                self.lifecycle.shutdown()
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"lifecycle: {cleanup_error}"
                )

            if cleanup_errors:
                self.last_error += (
                    "; cleanup: "
                    + "; ".join(cleanup_errors)
                )

            raise RuntimeError(
                f"betterVRAMd startup failed: {self.last_error}"
            ) from error

    def stop(self) -> None:
        """Stop components and release cached model mappings."""

        if self.state == ManagerState.STOPPED:
            # Still attempt lifecycle cleanup in case resources
            # were created outside normal manager startup.
            try:
                self.lifecycle.shutdown()
            except Exception as error:
                self.state = ManagerState.FAILED
                self.last_error = str(error)

                raise RuntimeError(
                    f"Shutdown encountered errors: {self.last_error}"
                ) from error

            return

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
        failed_components = []

        for name in reversed(self._started_components):
            component = self.components[name]

            try:
                component.stop()

            except Exception as error:
                errors.append(f"{name}: {error}")
                failed_components.append(name)

        # Preserve components whose shutdown failed.
        self._started_components = list(
            reversed(failed_components)
        )

        return errors

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of manager state."""
        lifecycle_status = self.lifecycle.get_status()

        return {
            "application": self.config.app_name,
            "version": self.config.version,
            "state": self.state.value,
            "registered_components": list(self.components.keys()),
            "active_components": list(self._started_components),
            "registered_models": len(self.model_registry.models),
            "cached_models": lifecycle_status["cached_models"],
            "active_model": lifecycle_status["active_model"],
            "worker_running": lifecycle_status["worker_running"],
            "scheduler": self.scheduler.get_status(),
            "last_error": self.last_error,
        }

    def get_resource_status(self) -> dict:
        """Collect passive system and GPU measurements."""

        if self.state != ManagerState.RUNNING:
            raise RuntimeError(
                "Resource monitoring requires a running manager."
            )

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

    def estimate_model_memory(
        self,
        name: str,
        gpu_layers: int = 0,
    ) -> dict:
        """Estimate model memory requirements without loading it."""

        metadata = self.lifecycle.inspect_model(name)
        budget = self.get_resource_budget()

        estimate = self.memory_estimator.estimate(
            metadata=metadata,
            budget=budget,
            gpu_layers=gpu_layers,
        )

        return self.memory_estimator.to_dict(estimate)

    def select_model_configuration(
        self,
        name: str,
        max_gpu_layers: int | None = None,
    ) -> dict:
        """Select the highest feasible GPU-layer configuration."""

        budget = self.get_resource_budget()

        if max_gpu_layers is None:
            max_gpu_layers = self.config.default_gpu_layers

        estimate = self.lifecycle.select_model_configuration(
            name=name,
            budget=budget,
            max_gpu_layers=max_gpu_layers,
        )

        return self.memory_estimator.to_dict(estimate)
