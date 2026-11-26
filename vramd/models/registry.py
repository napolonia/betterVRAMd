
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    name: str
    path: Path
    size_bytes: int
    format: str
    registered_at: float

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 ** 2), 2)


class ModelRegistry:
    """Maintain metadata for registered local models."""

    def __init__(self, model_directory: Path):
        self.model_directory = model_directory.expanduser()
        self.models: dict[str, ModelInfo] = {}

    def register(self, path: str | Path) -> ModelInfo:
        model_path = Path(path).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if model_path.suffix.lower() != ".gguf":
            raise ValueError("Only GGUF models are supported.")

        name = model_path.stem

        if name in self.models:
            raise ValueError(f"Model already registered: {name}")

        import time

        info = ModelInfo(
            name=name,
            path=model_path,
            size_bytes=model_path.stat().st_size,
            format="gguf",
            registered_at=time.time(),
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
        """Find GGUF files without automatically registering them."""

        if not self.model_directory.exists():
            return []

        return sorted(self.model_directory.glob("*.gguf"))
