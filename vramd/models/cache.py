
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CachedModel:
    name: str
    path: Path
    size_bytes: int
    mapping: mmap.mmap
    warmed: bool = False


class RAMCacheManager:
    """Manage read-only memory-mapped GGUF models."""

    def __init__(self):
        self._cache: dict[str, CachedModel] = {}

    
    def load(self, name: str, model_info) -> CachedModel:
        if name in self._cache:
            raise ValueError(f"Model already cached: {name}")

        path = Path(model_info.path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        size = path.stat().st_size

        if size == 0:
            raise ValueError(f"Empty model file: {path}")

        file_handle = os.open(path, os.O_RDONLY)

        try:
            mapping = mmap.mmap(
                file_handle,
                length=0,
                access=mmap.ACCESS_READ,
            )
        finally:
            os.close(file_handle)

        try:
            cached = CachedModel(
                name=name,
                path=path,
                size_bytes=size,
                mapping=mapping,
            )

            self._cache[name] = cached

            return cached

        except Exception:
            mapping.close()
            raise


    def warm(self, name: str) -> None:
        cached = self.get(name)

        # Read pages without creating a full-size bytes copy.
        page_size = mmap.PAGESIZE
        size = cached.size_bytes

        for offset in range(0, size, page_size):
            _ = cached.mapping[offset]

        cached.warmed = True

    def unload(self, name: str) -> None:
        cached = self.get(name)

        cached.mapping.close()
        del self._cache[name]

    def get(self, name: str) -> CachedModel:
        if name not in self._cache:
            raise KeyError(f"Model not cached: {name}")

        return self._cache[name]

    def list_cached(self) -> list[dict]:
        return [
            {
                "name": model.name,
                "path": str(model.path),
                "size_bytes": model.size_bytes,
                "warmed": model.warmed,
            }
            for model in self._cache.values()
        ]

    def clear(self) -> None:
        for name in list(self._cache):
            self.unload(name)
