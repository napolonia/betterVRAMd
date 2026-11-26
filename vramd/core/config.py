
import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"


class VRAMdConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Application
    app_name: str = "VRAMd"
    version: str = "0.1.0"
    log_level: str = "INFO"
    debug: bool = False

    # Paths
    project_root: Path = PROJECT_ROOT
    model_directory: Path = Path.home() / "models-tmp"
    cache_directory: Path = Path.home() / ".cache/vramd"
    log_directory: Path = Path.home() / ".local/share/vramd/logs"
    runtime_directory: Path = Path("/tmp/vramd")
    config_directory: Path = PROJECT_ROOT / "config"
    llama_cpp_directory: Path = Path.home() / "llama.cpp"
    llama_cli: Path = Path.home() / "llama.cpp/build-cuda/bin/llama-cli"
    llama_server: Path = Path.home() / "llama.cpp/build-cuda/bin/llama-server"

    # GPU
    gpu_enabled: bool = True
    device_index: int = 0
    max_vram_mb: int = Field(default=1800, ge=0)
    monitor_interval_seconds: float = Field(default=1.0, gt=0)
    verify_memory_release: bool = True
    automatic_budget: bool = True
    minimum_free_vram_mb: int = Field(default=512, ge=0)

    # System memory
    max_ram_mb: int = Field(default=4096, ge=512)
    memory_reserve_mb: int = Field(default=2048, ge=512)
    warm_cache_enabled: bool = True
    allow_memory_locking: bool = False
    cache_strategy: str = "mmap"

    # Inference
    default_backend: str = "llama.cpp"
    default_mode: str = "hybrid"
    max_workers: int = Field(default=1, ge=1, le=1)
    worker_timeout_seconds: int = Field(default=300, ge=1)
    context_size: int = Field(default=2048, ge=512)
    default_gpu_layers: int = Field(default=10, ge=0)
    threads: int = Field(default=4, ge=1)

    # Worker
    use_separate_process: bool = True
    shutdown_timeout_seconds: int = Field(default=10, ge=1)
    force_termination: bool = True
    restart_on_crash: bool = False

    # Scheduler
    scheduler_enabled: bool = True
    queue_enabled: bool = True
    max_queue_size: int = Field(default=10, ge=1)
    priority_enabled: bool = False

    # Registry
    registry_enabled: bool = True
    auto_discover: bool = False
    supported_formats: list[str] = ["gguf"]

    # API
    api_enabled: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8765, ge=1, le=65535)
    api_authentication: bool = False

    # Logging
    logging_enabled: bool = True
    file_logging: bool = True
    console_logging: bool = True
    max_log_file_size_mb: int = Field(default=10, ge=1)
    log_backup_count: int = Field(default=3, ge=0)

    @field_validator(
        "project_root", "model_directory", "cache_directory",
        "log_directory", "runtime_directory", "config_directory",
        "llama_cpp_directory", "llama_cli", "llama_server",
        mode="before",
    )
    @classmethod
    def expand_paths(cls, value):
        return Path(value).expanduser()


TOML_MAPPING = {
    "application": {
        "name": "app_name",
        "version": "version",
        "log_level": "log_level",
        "debug": "debug",
    },
    "paths": {
        "project_root": "project_root",
        "model_directory": "model_directory",
        "cache_directory": "cache_directory",
        "log_directory": "log_directory",
        "runtime_directory": "runtime_directory",
        "config_directory": "config_directory",
        "llama_cpp_directory": "llama_cpp_directory",
        "llama_cli": "llama_cli",
        "llama_server": "llama_server",
    },
    "gpu": {
        "enabled": "gpu_enabled",
        "device_index": "device_index",
        "max_vram_mb": "max_vram_mb",
        "monitor_interval_seconds": "monitor_interval_seconds",
        "verify_memory_release": "verify_memory_release",
        "automatic_budget": "automatic_budget",
        "minimum_free_vram_mb": "minimum_free_vram_mb",
    },
    "memory": {
        "max_ram_mb": "max_ram_mb",
        "reserve_mb": "memory_reserve_mb",
        "warm_cache_enabled": "warm_cache_enabled",
        "allow_memory_locking": "allow_memory_locking",
        "cache_strategy": "cache_strategy",
    },
    "inference": {
        "default_backend": "default_backend",
        "default_mode": "default_mode",
        "max_workers": "max_workers",
        "worker_timeout_seconds": "worker_timeout_seconds",
        "context_size": "context_size",
        "default_gpu_layers": "default_gpu_layers",
        "threads": "threads",
    },
    "worker": {
        "use_separate_process": "use_separate_process",
        "shutdown_timeout_seconds": "shutdown_timeout_seconds",
        "force_termination": "force_termination",
        "restart_on_crash": "restart_on_crash",
    },
    "scheduler": {
        "enabled": "scheduler_enabled",
        "queue_enabled": "queue_enabled",
        "max_queue_size": "max_queue_size",
        "priority_enabled": "priority_enabled",
    },
    "registry": {
        "enabled": "registry_enabled",
        "auto_discover": "auto_discover",
        "supported_formats": "supported_formats",
    },
    "api": {
        "enabled": "api_enabled",
        "host": "api_host",
        "port": "api_port",
        "authentication": "api_authentication",
    },
    "logging": {
        "enabled": "logging_enabled",
        "file_logging": "file_logging",
        "console_logging": "console_logging",
        "max_file_size_mb": "max_log_file_size_mb",
        "backup_count": "log_backup_count",
    },
}


def load_config(config_path: Path = DEFAULT_CONFIG) -> VRAMdConfig:
    """Load TOML configuration, then apply VRAMD_* environment overrides."""

    with config_path.open("rb") as file:
        toml_data = tomllib.load(file)

    values = {}

    for section, fields in TOML_MAPPING.items():
        section_data = toml_data.get(section, {})
        for toml_key, model_key in fields.items():
            if toml_key in section_data:
                values[model_key] = section_data[toml_key]

    # Environment variables take precedence over TOML.
    for field_name in VRAMdConfig.model_fields:
        env_name = f"VRAMD_{field_name.upper()}"
        if env_name in os.environ:
            values[field_name] = os.environ[env_name]

    return VRAMdConfig.model_validate(values)
