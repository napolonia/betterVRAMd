
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

    LAYER_PATTERN = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")

    @staticmethod
    def _get_layer_count(metadata: dict) -> int:
        architecture = metadata.get("architecture")

        if not architecture:
            return 0

        model_metadata = metadata.get("metadata", {})

        return int(
            model_metadata.get(
                f"{architecture}.block_count", 0
            )
        )

    def estimate(
        self,
        metadata: dict,
        budget: ResourceBudget,
        gpu_layers: int = 0,
    ) -> ModelMemoryEstimate:

        warnings = []

        tensor_bytes = metadata.get("tensor_bytes")

        if (
            not isinstance(tensor_bytes, (int, float))
            or tensor_bytes <= 0
        ):
            raise ValueError(
                "Valid tensor_bytes are required for estimation."
            )

        block_count = self._get_layer_count(metadata)

        if block_count < 0:
            raise ValueError("Invalid transformer block count.")

        # llama.cpp counts the output layer in -ngl.
        total_layers = block_count + 1

        if gpu_layers < 0:
            raise ValueError("gpu_layers cannot be negative.")

        if gpu_layers > total_layers:
            raise ValueError(
                f"Requested {gpu_layers} GPU layers, "
                f"but llama.cpp supports at most {total_layers} "
                "offload layers for this model."
            )

        gpu_bytes = 0
        total_tensor_bytes = int(tensor_bytes)

        if gpu_layers > 0:
            tensors = metadata.get("tensors")

            if not isinstance(tensors, list) or not tensors:
                raise ValueError(
                    "Full tensor metadata is required for "
                    "GPU-layer estimation."
                )

            output_bytes = 0
            layer_bytes = {}
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
                        f"Invalid tensor size: {name}"
                    )

                inspected_tensor_bytes += size_bytes

                # llama.cpp output-layer tensors.
                if name.startswith("output."):
                    output_bytes += size_bytes
                    continue

                match = self.LAYER_PATTERN.search(name)

                if match:
                    layer_index = int(match.group(1))

                    if layer_index >= block_count:
                        raise ValueError(
                            f"Tensor {name} references an invalid "
                            "transformer block."
                        )

                    layer_bytes[layer_index] = (
                        layer_bytes.get(layer_index, 0)
                        + size_bytes
                    )

            if inspected_tensor_bytes <= 0:
                raise ValueError(
                    "No measurable tensor data found."
                )

            if inspected_tensor_bytes != total_tensor_bytes:
                warnings.append(
                    "Tensor byte total differs from aggregate "
                    "metadata. Using the inspected tensor total."
                )

                total_tensor_bytes = inspected_tensor_bytes

            if output_bytes <= 0:
                raise ValueError(
                    "Output-layer tensors could not be identified. "
                    "GPU estimation cannot safely proceed."
                )

            # -ngl 1 includes the output layer.
            gpu_bytes = output_bytes

            # Additional -ngl values select transformer blocks
            # starting from the final block.
            selected_block_count = gpu_layers - 1

            if selected_block_count > 0:
                first_gpu_block = (
                    block_count - selected_block_count
                )

                selected_blocks = range(
                    first_gpu_block,
                    block_count,
                )

                for block_index in selected_blocks:
                    if block_index not in layer_bytes:
                        raise ValueError(
                            f"Missing tensor information for "
                            f"transformer block {block_index}."
                        )

                    gpu_bytes += layer_bytes[block_index]

            warnings.append(
                "Non-layer tensors remain assigned to RAM. "
                "Actual llama.cpp backend placement may differ."
            )

            warnings.append(
                "Output-layer identification uses GGUF tensor names. "
                "Architecture-specific tensor placement may differ."
            )

        ram_bytes = max(
            0,
            total_tensor_bytes - gpu_bytes,
        )

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
