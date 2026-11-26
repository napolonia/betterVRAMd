
from dataclasses import dataclass, asdict

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

    def estimate(
        self,
        metadata: dict,
        budget: ResourceBudget,
        gpu_layers: int = 0,
    ) -> ModelMemoryEstimate:

        warnings = []

        tensor_bytes = metadata.get("tensor_bytes")

        if tensor_bytes is None or tensor_bytes <= 0:
            raise ValueError(
                "Valid tensor_bytes are required for estimation."
            )

        architecture = metadata.get("architecture")
        total_layers = 0

        if architecture:
            total_layers = int(
                metadata["metadata"].get(
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

        if gpu_layers == 0:
            gpu_bytes = 0
            mode = "cpu"

        else:
            # Estimate the offloaded portion proportionally.
            gpu_bytes = tensor_bytes * (
                gpu_layers / total_layers
            )
            mode = (
                "gpu" if gpu_layers == total_layers
                else "hybrid"
            )

            warnings.append(
                "GPU memory is estimated proportionally by layer "
                "count; actual tensor placement may differ."
            )

        ram_bytes = tensor_bytes - gpu_bytes

        estimated_ram_mb = (
            ram_bytes * self.SAFETY_MARGIN / (1024 ** 2)
        )

        estimated_vram_mb = (
            gpu_bytes * self.SAFETY_MARGIN / (1024 ** 2)
        )

        # KV cache, context buffers and backend overhead are not
        # included in the tensor storage estimate.
        warnings.append(
            "KV cache and inference runtime overhead are not included."
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
