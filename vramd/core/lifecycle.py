
from vramd.models.cache import RAMCacheManager
from vramd.models.metadata import ModelMetadataInspector
from vramd.models.registry import ModelInfo, ModelRegistry


class LifecycleController:
    """Coordinate registered models, metadata and RAM caching."""

    def __init__(
        self,
        registry: ModelRegistry,
        cache: RAMCacheManager,
        metadata_inspector: ModelMetadataInspector,
    ):
        self.registry = registry
        self.cache = cache
        self.metadata_inspector = metadata_inspector

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

    def cache_model(self, name: str):
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
        self.cache.clear()
