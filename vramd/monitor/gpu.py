
from dataclasses import dataclass, asdict

import pynvml


@dataclass
class GPUMemorySnapshot:
    device_name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
    gpu_utilization_percent: int
    memory_utilization_percent: int
    temperature_c: int
    power_state: int


class GPUMonitor:
    """Read-only NVIDIA GPU monitoring through NVML."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.handle = None
        self.initialized = False

    @staticmethod
    def _to_mb(value: int) -> float:
        return round(value / (1024 ** 2), 2)

    def start(self) -> None:
        if self.initialized:
            return

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(
            self.device_index
        )
        self.initialized = True

    def get_snapshot(self) -> GPUMemorySnapshot:
        if not self.initialized or self.handle is None:
            raise RuntimeError("GPU monitor is not initialized.")

        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(
            self.handle
        )
        temperature = pynvml.nvmlDeviceGetTemperature(
            self.handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        power_state = pynvml.nvmlDeviceGetPowerState(self.handle)

        name = pynvml.nvmlDeviceGetName(self.handle)

        if isinstance(name, bytes):
            name = name.decode("utf-8")

        return GPUMemorySnapshot(
            device_name=name,
            total_vram_mb=self._to_mb(memory.total),
            used_vram_mb=self._to_mb(memory.used),
            free_vram_mb=self._to_mb(memory.free),
            gpu_utilization_percent=utilization.gpu,
            memory_utilization_percent=utilization.memory,
            temperature_c=temperature,
            power_state=power_state,
        )

    def get_status(self) -> dict:
        return asdict(self.get_snapshot())

    def stop(self) -> None:
        if self.initialized:
            pynvml.nvmlShutdown()
            self.handle = None
            self.initialized = False
