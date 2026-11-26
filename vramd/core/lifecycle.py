
from vramd.core.budget import ResourceBudget
from vramd.models.cache import RAMCacheManager
from vramd.models.metadata import ModelMetadataInspector
from vramd.models.registry import ModelInfo, ModelRegistry
from vramd.models.estimator import ModelMemoryEstimate
from vramd.models.selector import ModelLayerSelector


class LifecycleController:
    """Coordinate registered models, metadata and RAM caching."""

    def __init__(
        self,
        registry: ModelRegistry,
        cache: RAMCacheManager,
        metadata_inspector: ModelMetadataInspector,
        layer_selector: ModelLayerSelector | None = None,
    ):
        self.registry = registry
        self.cache = cache
        self.metadata_inspector = metadata_inspector
        self.layer_selector = layer_selector

    def register_model(self, path: str) -> ModelInfo:
        return self.registry.register(path)

    def unregister_model(self, name: str) -> None:
        self.registry.get(name)

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        if name in cached_names:
            self.cache.unload(name)

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
        """Recommend a feasible CPU/GPU configuration without loading."""

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
        """Create a read-only memory mapping for a registered model."""

        model = self.registry.get(name)

        return self.cache.load(name, model)

    def unload_model(self, name: str) -> None:
        self.cache.unload(name)

    def warm_model(self, name: str) -> None:
        self.cache.warm(name)

    def get_model_state(self, name: str) -> str:
        self.registry.get(name)

        cached_names = {
            item["name"]
            for item in self.cache.list_cached()
        }

        return (
            "cached" if name in cached_names
            else "registered"
        )

    def list_models(self) -> list[dict]:
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
                    "cached"
                    if model.name in cached_names
                    else "registered"
                ),
            }
            for model in self.registry.models.values()
        ]

    def shutdown(self) -> None:
        """Release all active memory mappings."""

        self.cache.clear()
