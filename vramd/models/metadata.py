
from pathlib import Path

from gguf import GGUFReader

from vramd.core.config import VRAMdConfig


class ModelMetadataError(Exception):
    """Raised when GGUF metadata inspection fails."""


class ModelMetadataInspector:
    """Inspect GGUF metadata using llama.cpp's GGUF reader."""

    def __init__(self, config: VRAMdConfig):
        self.config = config

    @staticmethod
    def _decode_value(field):
        value = field.contents()

        if hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "item"):
            value = value.item()

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        return value

    def inspect(self, path: str | Path) -> dict:
        model_path = Path(path).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if model_path.suffix.lower() != ".gguf":
            raise ModelMetadataError(
                "Only GGUF files are supported."
            )

        try:
            reader = GGUFReader(str(model_path))

            metadata = {}

            for name, field in reader.fields.items():
                try:
                    metadata[name] = self._decode_value(field)
                except (ValueError, IndexError, TypeError):
                    continue

            architecture = metadata.get("general.architecture")

            context_length = None

            if architecture:
                context_length = metadata.get(
                    f"{architecture}.context_length"
                )

            tensors = []
            total_parameters = 0
            quantization_types = {}
            total_tensor_bytes = 0

            for tensor in reader.tensors:
                shape = [
                    int(dimension)
                    for dimension in tensor.shape
                ]

                tensor_type = tensor.tensor_type.name
                tensor_bytes = int(tensor.n_bytes)
                parameters = int(tensor.n_elements)

                total_parameters += parameters
                total_tensor_bytes += tensor_bytes

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
                    total_parameters / 1_000_000_000,
                    3,
                ),
                "quantization_types": quantization_types,
                "tensor_bytes": total_tensor_bytes,
                "tensor_size_mb": round(
                    total_tensor_bytes / (1024 ** 2),
                    2,
                ),
            }

        except ModelMetadataError:
            raise
        except Exception as error:
            raise ModelMetadataError(
                f"Failed to inspect GGUF model "
                f"{model_path}: {error}"
            ) from error
