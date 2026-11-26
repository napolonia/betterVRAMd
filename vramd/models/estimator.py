
import re
from dataclasses import asdict, dataclass

from vramd.core.budget import ResourceBudget


@dataclass
class ModelMemoryEstimate:
    mode: str
    gpu_layers: int
    total_layers: int
    estimated_ram_mb: float
    estimated_vram_mb: float
    available_ram_mb: float
    available_vram_mb: float
    feasible: bool
    warnings: list[str]


class ModelMemoryEstimator:
    """Estimate model memory requirements without allocating resources."""

    SAFETY_MARGIN = 1.15

    # Standard llama.cpp transformer block naming convention.
    LAYER_PATTERN = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")

    def estimate(
        self,
        metadata: dict,
        budget: ResourceBudget,
        gpu_layers: int = 0,
    ) -> ModelMemoryEstimate:

        warnings = []

        tensor_bytes = metadata.get("tensor_bytes")

        if not isinstance(tensor_bytes, (int, float)) or tensor_bytes <= 0:
            raise ValueError(
                "Valid tensor_bytes are required for estimation."
            )

        architecture = metadata.get("architecture")
        model_metadata = metadata.get("metadata", {})

        total_layers = 0

        if architecture:
            total_layers = int(
                model_metadata.get(
                    f"{architecture}.block_count", 0
                )
            )

        if gpu_layers < 0:
            raise ValueError("gpu_layers cannot be negative.")

        if gpu_layers > total_layers:
            raise ValueError(
                f"Requested {gpu_layers} GPU layers, "
                f"but model has {total_layers} layers."
            )

        if gpu_layers > 0 and total_layers == 0:
            raise ValueError(
                "Layer count unavailable; GPU estimation is unsafe."
            )

        total_tensor_bytes = int(tensor_bytes)
        gpu_bytes = 0

        if gpu_layers > 0:
            tensors = metadata.get("tensors")

            if not isinstance(tensors, list) or not tensors:
                raise ValueError(
                    "Full tensor metadata is required for "
                    "layer-based GPU estimation."
                )

            first_gpu_layer = total_layers - gpu_layers

            selected_layers = set(
                range(first_gpu_layer, total_layers)
            )

            found_layers = set()
            inspected_tensor_bytes = 0

            for tensor in tensors:
                try:
                    name = str(tensor["name"])
                    size_bytes = int(tensor["size_bytes"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        "Invalid tensor metadata encountered."
                    )

                if size_bytes < 0:
                    raise ValueError(
                        f"Invalid tensor size for {name}."
                    )

                inspected_tensor_bytes += size_bytes

                match = self.LAYER_PATTERN.search(name)

                if match is None:
                    continue

                layer_index = int(match.group(1))

                if layer_index >= total_layers:
                    warnings.append(
                        f"Tensor {name} references a layer "
                        "outside the declared architecture."
                    )
                    continue

                found_layers.add(layer_index)

                if layer_index in selected_layers:
                    gpu_bytes += size_bytes

            if inspected_tensor_bytes == 0:
                raise ValueError(
                    "Tensor metadata contains no measurable bytes."
                )

            if inspected_tensor_bytes != total_tensor_bytes:
                warnings.append(
                    "Tensor byte totals differ from the aggregate "
                    "metadata; using the larger value conservatively."
                )

                total_tensor_bytes = max(
                    total_tensor_bytes,
                    inspected_tensor_bytes,
                )

            missing_layers = selected_layers - found_layers

            if missing_layers:
                raise ValueError(
                    "Unable to account for selected GPU layers: "
                    f"{sorted(missing_layers)}"
                )

            warnings.append(
                "Non-layer tensors remain assigned to RAM. "
                "Actual llama.cpp tensor placement may differ."
            )

        ram_bytes = max(0, total_tensor_bytes - gpu_bytes)

        estimated_ram_mb = (
            ram_bytes * self.SAFETY_MARGIN / (1024 ** 2)
        )

        estimated_vram_mb = (
            gpu_bytes * self.SAFETY_MARGIN / (1024 ** 2)
        )

        if gpu_layers == 0:
            mode = "cpu"
        elif gpu_layers == total_layers:
            mode = "gpu"
        else:
            mode = "hybrid"

        warnings.append(
            "KV cache, computation buffers and inference runtime "
            "overhead are not included."
        )

        feasible = (
            estimated_ram_mb <= budget.allocatable_ram_mb
            and (
                gpu_layers == 0
                or (
                    budget.gpu_allowed
                    and estimated_vram_mb
                    <= budget.allocatable_vram_mb
                )
            )
        )

        if estimated_ram_mb > budget.allocatable_ram_mb:
            warnings.append("Insufficient safe RAM budget.")

        if (
            gpu_layers > 0
            and (
                not budget.gpu_allowed
                or estimated_vram_mb > budget.allocatable_vram_mb
            )
        ):
            warnings.append("Insufficient safe VRAM budget.")

        return ModelMemoryEstimate(
            mode=mode,
            gpu_layers=gpu_layers,
            total_layers=total_layers,
            estimated_ram_mb=round(estimated_ram_mb, 2),
            estimated_vram_mb=round(estimated_vram_mb, 2),
            available_ram_mb=budget.allocatable_ram_mb,
            available_vram_mb=budget.allocatable_vram_mb,
            feasible=feasible,
            warnings=warnings,
        )

    @staticmethod
    def to_dict(
        estimate: ModelMemoryEstimate,
    ) -> dict:
        return asdict(estimate)
