
from datetime import datetime

import pynvml


def main():
    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name = pynvml.nvmlDeviceGetName(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            process_memory = sum(
                p.usedGpuMemory for p in processes
                if p.usedGpuMemory is not None
            )
        except pynvml.NVMLError:
            processes = []
            process_memory = None

        print("\n=== VRAMd GPU Probe ===")
        print(f"Timestamp:          {datetime.now().isoformat(timespec='seconds')}")
        print(f"GPU:                {name}")
        print(f"Total VRAM:         {memory.total / 1024**2:.0f} MiB")
        print(f"Used VRAM:          {memory.used / 1024**2:.0f} MiB")
        print(f"Free VRAM:          {memory.free / 1024**2:.0f} MiB")
        print(f"GPU Usage:          {utilization.gpu}%")
        print(f"Memory Utilization: {utilization.memory}%")
        print(f"Visible processes:  {len(processes)}")

        if process_memory is not None:
            print(f"Process VRAM sum:   {process_memory / 1024**2:.0f} MiB")
        else:
            print("Process VRAM sum:   Unavailable")

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
