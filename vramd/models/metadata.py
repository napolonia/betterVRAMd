
from pathlib import Path
import sys

from vramd.core.config import VRAMdConfig


class ModelMetadataError(Exception):
    """Raised when GGUF metadata inspection fails."""


class ModelMetadataInspector:
    """Inspect GGUF metadata using the configured llama.cpp tooling."""

    def __init__(self, config: VRAMdConfig):
        self.config = config
        self._reader_class = None

    def _load_reader(self):
        if self._reader_class is not None:
            return self._reader_class

        gguf_path = (
            self.config.llama_cpp_directory / "gguf-py"
        ).expanduser().resolve()

        if not (gguf_path / "gguf" / "gguf_reader.py").is_file():
            raise ModelMetadataError(
                f"GGUF Python package not found: {gguf_path}"
            )

        path_string = str(gguf_path)

        if path_string not in sys.path:
            sys.path.insert(0, path_string)

        try:
            from gguf import GGUFReader
        except ImportError as error:
            raise ModelMetadataError(
                f"Unable to import llama.cpp GGUF reader: {error}"
            ) from error

        self._reader_class = GGUFReader
        return self._reader_class

    @staticmethod
    def _decode_value(field):
        value = field.contents()

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, list):
            if len(value) == 1:
                return value[0]

            return value

        return value

    def inspect(self, path: str | Path) -> dict:
        model_path = Path(path).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if model_path.suffix.lower() != ".gguf":
            raise ModelMetadataError("Only GGUF files are supported.")

        try:
            reader_class = self._load_reader()
            reader = reader_class(str(model_path))

            metadata = {}

            for name, field in reader.fields.items():
                try:
                    metadata[name] = self._decode_value(field)
                except (ValueError, IndexError, TypeError):
                    continue

            tensors = []

            total_parameters = 0
            quantization_types = {}
            total_tensor_bytes = 0

            for tensor in reader.tensors:
                shape = tensor.shape.tolist()
                tensor_type = str(tensor.tensor_type)
                tensor_bytes = int(tensor.n_bytes)
                total_tensor_bytes += tensor_bytes

                parameters = 1
                for dimension in shape:
                    parameters *= int(dimension)

                total_parameters += parameters
                quantization_types[tensor_type] = (
                    quantization_types.get(tensor_type, 0) + 1
                )

                tensors.append({
                    "name": tensor.name,
                    "shape": shape,
                    "type": tensor_type,
                    "parameters": parameters,
                    "size_bytes": tensor_bytes,
                })

            architecture = metadata.get(
                "general.architecture"
            )

            context_length = None

            if architecture:
                context_length = metadata.get(
                    f"{architecture}.context_length"
                )

            return {
                "name": model_path.stem,
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "architecture": architecture,
                "context_length": context_length,
                "metadata": metadata,
                "tensor_count": len(tensors),
                "tensors": tensors,
                "parameter_count": total_parameters,
                "parameter_count_billions": round(
                    total_parameters / 1_000_000_000, 3
                ),
                "quantization_types": quantization_types,
                "tensor_bytes": total_tensor_bytes,
                "tensor_size_mb": round(
                    total_tensor_bytes / (1024 ** 2), 2
                ),
            }

        except ModelMetadataError:
            raise
        except Exception as error:
            raise ModelMetadataError(
                f"Failed to inspect GGUF model {model_path}: {error}"
            ) from error
