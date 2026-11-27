
"""Coordinate model registration, caching and inference lifecycle."""

from vramd.core.budget import ResourceBudget
from vramd.core.config import VRAMdConfig
from vramd.models.cache import RAMCacheManager
from vramd.models.estimator import ModelMemoryEstimate
from vramd.models.metadata import ModelMetadataInspector
from vramd.models.registry import ModelInfo, ModelRegistry
from vramd.models.selector import ModelLayerSelector
from vramd.worker.process import WorkerProcess


class LifecycleController:
    """Coordinate models, RAM caching and inference processes."""

    def __init__(
        self,
        registry: ModelRegistry,
        cache: RAMCacheManager,
        metadata_inspector: ModelMetadataInspector,
        layer_selector: ModelLayerSelector | None = None,
        config: VRAMdConfig | None = None,
    ):
        self.registry = registry
        self.cache = cache
        self.metadata_inspector = metadata_inspector
        self.layer_selector = layer_selector
        self.config = config

        self.worker = (
            WorkerProcess(config)
            if config is not None
            else None
        )

        self._active_model: str | None = None

    @property
    def active_model(self) -> str | None:
        """Return the currently active model name."""
        return self._active_model

    @property
    def backend(self):
        """Return the active inference backend, if available."""
        return self.worker.backend if self.worker else None

    def register_model(self, path: str) -> ModelInfo:
        """Register a model without loading its weights."""
        return self.registry.register(path)

    def unregister_model(self, name: str) -> None:
        """Unregister a model after releasing any cached resources."""

        self.registry.get(name)

        if name == self._active_model:
            self.unload_model(name)
        else:
            cached_names = {
                item["name"]
                for item in self.cache.list_cached()
            }

            if name in cached_names:
                self.unload_model(name)

        self.registry.unregister(name)

    def inspect_model(self, name: str) -> dict:
        """Inspect GGUF metadata without loading model weights."""

        model = self.registry.get(name)

        return self.metadata_inspector.inspect(model.path)

    def select_model_configuration(
        self,
        name: str,
        budget: ResourceBudget,
        max_gpu_layers: int | None = None,
    ) -> ModelMemoryEstimate:
        """Select a feasible CPU/GPU configuration without loading."""

        if self.layer_selector is None:
            raise RuntimeError(
                "ModelLayerSelector has not been configured."
            )

        metadata = self.inspect_model(name)

        return self.layer_selector.select(
            metadata=metadata,
            budget=budget,
            max_gpu_layers=max_gpu_layers,
        )

    def cache_model(self, name: str):
        """Create a read-only memory mapping for one registered model."""

        model = self.registry.get(name)

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        if cached_names and name not in cached_names:
            raise RuntimeError(
                "Only one model may be cached at a time. "
                "Unload the existing model first."
            )

        if self.worker is not None and self.worker.is_running:
            if name != self._active_model:
                raise RuntimeError(
                    "Cannot cache another model while an "
                    "inference worker is active."
                )

        return self.cache.load(name, model)

    def load_model(
        self,
        name: str,
        budget: ResourceBudget,
        max_gpu_layers: int | None = None,
    ):
        """
        Select, cache and start one model.

        Resource evaluation must succeed before model caching
        or worker startup.

        The caller must supply a current resource budget.
        """

        if self.worker is None:
            raise RuntimeError(
                "WorkerProcess requires a configuration."
            )

        if self._active_model is not None:
            raise RuntimeError(
                f"Model '{self._active_model}' is already active. "
                "Unload it before loading another model."
            )

        if self.worker.is_running:
            raise RuntimeError(
                "An inference worker is already running."
            )

        # Clear any stale, already-terminated worker reference.
        if self.worker.process is not None:
            self.worker.stop()

        model = self.registry.get(name)

        # Evaluate resource requirements before caching or loading.
        if max_gpu_layers is None:
            max_gpu_layers = self.config.default_gpu_layers

        estimate = self.select_model_configuration(
            name=name,
            budget=budget,
            max_gpu_layers=max_gpu_layers,
        )

        if not estimate.feasible:
            raise RuntimeError(
                "Selected model configuration is not feasible "
                "under the current resource budget."
            )

        selected_gpu_layers = estimate.gpu_layers

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        if cached_names and name not in cached_names:
            raise RuntimeError(
                "Another model is cached. Unload it first."
            )

        cache_created = False

        if name not in cached_names:
            self.cache_model(name)
            cache_created = True

        # Reserve the active-model identity before starting the worker.
        self._active_model = name

        try:
            backend = self.worker.start(
                model_path=model.path,
                gpu_layers=selected_gpu_layers,
                mmproj_path=model.mmproj_path,
            )

            if not self.worker.is_running:
                raise RuntimeError(
                    "Inference worker stopped immediately "
                    "after becoming ready."
                )

            return backend

        except Exception as exc:

            # A failed startup may leave a process tracked if cleanup
            # could not terminate it. Preserve its model mapping.
            if (
                self.worker.process is not None
                and not self.worker.is_running
            ):
                try:
                    self.worker.stop()
                except Exception:
                    pass

            if self.worker.process is not None:
                raise RuntimeError(
                    f"Model '{name}' startup failed and the worker "
                    "process remains tracked. The model mapping "
                    "has been retained for safe cleanup."
                ) from exc

            self._active_model = None

            # Only release a mapping created by this load attempt.
            # Preserve mappings that existed before the attempt.
            if cache_created:
                try:
                    self.cache.unload(name)
                except Exception as cleanup_error:
                    raise RuntimeError(
                        f"Model '{name}' startup failed and its "
                        f"cache cleanup also failed: {cleanup_error}"
                    ) from exc

            raise

    def unload_model(self, name: str) -> None:
        """Stop inference before releasing the model mapping."""

        self.registry.get(name)

        if self._active_model is not None:
            if name != self._active_model:
                raise RuntimeError(
                    f"Cannot unload '{name}': "
                    f"'{self._active_model}' is active."
                )

            if self.worker is None:
                raise RuntimeError(
                    "Active model has no associated worker."
                )

            # Do not release the mapping unless worker shutdown succeeds.
            self.worker.stop()

            self._active_model = None

        elif self.worker is not None and self.worker.is_running:
            raise RuntimeError(
                "An inference worker is running without a tracked "
                "active model. Cache unloading is unsafe."
            )

        elif (
            self.worker is not None
            and self.worker.process is not None
        ):
            # Clear a terminated but still tracked subprocess.
            self.worker.stop()

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        if name in cached_names:
            self.cache.unload(name)

    def warm_model(self, name: str) -> None:
        """Warm the cached model if warming is explicitly enabled."""

        if (
            self.config is not None
            and not self.config.warm_cache_enabled
        ):
            return

        self.cache.warm(name)

    def get_model_state(self, name: str) -> str:
        """Return the current model residency state."""

        self.registry.get(name)

        if name == self._active_model:
            return "running"

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        return (
            "cached" if name in cached_names
            else "registered"
        )

    def list_models(self) -> list[dict]:
        """List registered models and their current states."""

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        return [
            {
                "name": model.name,
                "path": str(model.path),
                "size_bytes": model.size_bytes,
                "size_mb": model.size_mb,
                "state": (
                    "running"
                    if model.name == self._active_model
                    else "cached"
                    if model.name in cached_names
                    else "registered"
                ),
            }
            for model in self.registry.models.values()
        ]

    def shutdown(self) -> None:
        """Stop inference before releasing all memory mappings."""

        if self.worker is not None:
            if (
                self.worker.process is not None
                or self._active_model is not None
            ):
                self.worker.stop()

        elif self._active_model is not None:
            raise RuntimeError(
                "Active model has no associated worker. "
                "Cannot safely release its resources."
            )

        self._active_model = None

        # Only reached after successful worker shutdown.
        self.cache.clear()
