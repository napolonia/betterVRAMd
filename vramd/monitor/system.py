
from dataclasses import dataclass, asdict

import psutil


@dataclass
class SystemMemorySnapshot:
    total_ram_mb: float
    available_ram_mb: float
    used_ram_mb: float
    ram_percent: float

    total_swap_mb: float
    used_swap_mb: float
    available_swap_mb: float
    swap_percent: float

    process_ram_mb: float


class SystemMonitor:
    """Monitor system and current-process memory usage."""

    def __init__(self):
        self.process = psutil.Process()

    @staticmethod
    def _to_mb(value: int) -> float:
        return round(value / (1024 ** 2), 2)

    def get_snapshot(self) -> SystemMemorySnapshot:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process_memory = self.process.memory_info()

        return SystemMemorySnapshot(
            total_ram_mb=self._to_mb(memory.total),
            available_ram_mb=self._to_mb(memory.available),
            used_ram_mb=self._to_mb(memory.used),
            ram_percent=memory.percent,

            total_swap_mb=self._to_mb(swap.total),
            used_swap_mb=self._to_mb(swap.used),
            available_swap_mb=self._to_mb(
                swap.total - swap.used
            ),
            swap_percent=swap.percent,

            process_ram_mb=self._to_mb(process_memory.rss),
        )

    def get_status(self) -> dict:
        return asdict(self.get_snapshot())

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
