
from dataclasses import dataclass

from vramd.core.config import VRAMdConfig


@dataclass
class ResourceBudget:
    total_vram_mb: float
    used_vram_mb: float
    available_vram_mb: float
    allocatable_vram_mb: float

    total_ram_mb: float
    available_ram_mb: float
    allocatable_ram_mb: float

    gpu_allowed: bool
    warnings: list[str]


class ResourceBudgetCalculator:
    """Calculate safe resource limits without allocating memory."""

    def __init__(self, config: VRAMdConfig):
        self.config = config

    def calculate(
        self,
        system: dict,
        gpu: dict | None,
    ) -> ResourceBudget:

        warnings = []

        # RAM budget
        available_ram = system["available_ram_mb"]

        allocatable_ram = max(
            0,
            min(
                self.config.max_ram_mb,
                available_ram - self.config.memory_reserve_mb,
            ),
        )

        if allocatable_ram == 0:
            warnings.append("Insufficient available system RAM.")

        # GPU budget
        if gpu is None:
            return ResourceBudget(
                total_vram_mb=0,
                used_vram_mb=0,
                available_vram_mb=0,
                allocatable_vram_mb=0,
                total_ram_mb=system["total_ram_mb"],
                available_ram_mb=available_ram,
                allocatable_ram_mb=allocatable_ram,
                gpu_allowed=False,
                warnings=warnings + ["GPU unavailable or disabled."],
            )

        total_vram = gpu["total_vram_mb"]
        used_vram = gpu["used_vram_mb"]
        free_vram = gpu["free_vram_mb"]

        allocatable_vram = max(
            0,
            min(
                self.config.max_vram_mb,
                free_vram - self.config.minimum_free_vram_mb,
            ),
        )

        gpu_allowed = (
            allocatable_vram > 0
            and self.config.gpu_enabled
        )

        if not gpu_allowed:
            warnings.append("Insufficient safe GPU memory.")

        return ResourceBudget(
            total_vram_mb=total_vram,
            used_vram_mb=used_vram,
            available_vram_mb=free_vram,
            allocatable_vram_mb=allocatable_vram,
            total_ram_mb=system["total_ram_mb"],
            available_ram_mb=available_ram,
            allocatable_ram_mb=allocatable_ram,
            gpu_allowed=gpu_allowed,
            warnings=warnings,
        )
