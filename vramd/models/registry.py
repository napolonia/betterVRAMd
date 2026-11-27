
"""Maintain metadata for registered local GGUF models."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

#One important note: This uses filename-prefix matching. It supports your current naming convention,
#  but we should later verify that the projector association is unambiguous.

@dataclass
class ModelInfo:
    name: str
    path: Path
    size_bytes: int
    format: str
    registered_at: float
    mmproj_path: Path | None = None

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 ** 2), 2)


class ModelRegistry:
    """Maintain metadata for registered local models."""

    def __init__(self, model_directory: Path):
        self.model_directory = model_directory.expanduser()
        self.models: dict[str, ModelInfo] = {}

    def _find_projector(self, model_path: Path) -> Path | None:
        """Find a matching multimodal projector, if available."""

        prefix = f"mmproj-{model_path.stem}"

        candidates = sorted(
            model_path.parent.glob(f"{prefix}*.gguf")
        )

        return candidates[0].resolve() if candidates else None

    def register(self, path: str | Path) -> ModelInfo:
        model_path = Path(path).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if model_path.suffix.lower() != ".gguf":
            raise ValueError("Only GGUF models are supported.")

        if model_path.name.lower().startswith("mmproj-"):
            raise ValueError(
                "Multimodal projectors cannot be registered "
                "as independent models."
            )

        name = model_path.stem

        if name in self.models:
            raise ValueError(f"Model already registered: {name}")

        mmproj_path = self._find_projector(model_path)

        info = ModelInfo(
            name=name,
            path=model_path,
            size_bytes=model_path.stat().st_size,
            format="gguf",
            registered_at=time.time(),
            mmproj_path=mmproj_path,
        )

        self.models[name] = info
        return info

    def unregister(self, name: str) -> None:
        if name not in self.models:
            raise KeyError(f"Unknown model: {name}")

        del self.models[name]

    def get(self, name: str) -> ModelInfo:
        if name not in self.models:
            raise KeyError(f"Unknown model: {name}")

        return self.models[name]

    def list_models(self) -> list[dict]:
        return [
            asdict(model)
            for model in self.models.values()
        ]

    def discover(self) -> list[Path]:
        """Find main GGUF files without registering them."""

        if not self.model_directory.exists():
            return []

        return sorted(
            path
            for path in self.model_directory.glob("*.gguf")
            if not path.name.lower().startswith("mmproj-")
        )
