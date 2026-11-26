
from vramd.core.budget import ResourceBudget
from vramd.models.estimator import (
    ModelMemoryEstimate,
    ModelMemoryEstimator,
)


class NoFeasibleConfigurationError(RuntimeError):
    """Raised when no GPU/CPU configuration fits the resource budget."""

    def __init__(self, cpu_estimate: ModelMemoryEstimate):
        self.cpu_estimate = cpu_estimate

        super().__init__(
            "No feasible model configuration found. "
            "Even CPU-only estimation exceeds the safe RAM budget."
        )


class ModelLayerSelector:
    """Select the highest feasible llama.cpp GPU-layer count."""

    def __init__(
        self,
        estimator: ModelMemoryEstimator | None = None,
    ):
        self.estimator = estimator or ModelMemoryEstimator()

    def select(
        self,
        metadata: dict,
        budget: ResourceBudget,
        max_gpu_layers: int | None = None,
    ) -> ModelMemoryEstimate:
        """
        Find the highest feasible GPU-layer configuration.

        max_gpu_layers optionally limits the search range.
        Returns a ModelMemoryEstimate for the selected configuration.
        """

        if max_gpu_layers is not None and max_gpu_layers < 0:
            raise ValueError(
                "max_gpu_layers cannot be negative."
            )

        # Obtain the supported -ngl range.
        cpu_estimate = self.estimator.estimate(
            metadata,
            budget,
            gpu_layers=0,
        )

        total_layers = cpu_estimate.total_layers

        architecture = metadata.get("architecture")
        model_metadata = metadata.get("metadata", {})

        block_count = (
            model_metadata.get(
                f"{architecture}.block_count", 0
            )
            if architecture
            else 0
        )

        # Without reliable layer metadata, use CPU-only estimation.
        if not block_count or not budget.gpu_allowed:
            if cpu_estimate.feasible:
                if not budget.gpu_allowed:
                    cpu_estimate.warnings.append(
                        "GPU offloading unavailable; "
                        "selected CPU-only configuration."
                    )
                else:
                    cpu_estimate.warnings.append(
                        "Transformer block count unavailable; "
                        "GPU offloading disabled."
                    )

                return cpu_estimate

            raise NoFeasibleConfigurationError(cpu_estimate)

        search_limit = total_layers

        if max_gpu_layers is not None:
            search_limit = min(
                search_limit,
                max_gpu_layers,
            )

        # Search from the highest GPU-layer count downwards.
        # Do not assume feasibility is monotonic.
        for gpu_layers in range(search_limit, 0, -1):
            estimate = self.estimator.estimate(
                metadata,
                budget,
                gpu_layers=gpu_layers,
            )

            if estimate.feasible:
                if search_limit < total_layers:
                    estimate.warnings.append(
                        "GPU-layer selection was limited by "
                        "max_gpu_layers."
                    )

                return estimate

        # GPU configurations failed: try CPU-only.
        if cpu_estimate.feasible:
            cpu_estimate.warnings.append(
                "No GPU configuration was feasible; "
                "selected CPU-only operation."
            )

            return cpu_estimate

        raise NoFeasibleConfigurationError(cpu_estimate)
