"""Unit tests for VRAMdManager model lifecycle delegation.

These tests use mocks only; they do not start a worker or load a model.
"""

from unittest.mock import Mock

import pytest

from vramd.core.manager import ManagerState, VRAMdManager


@pytest.fixture
def manager():
    """Create a manager shell without initializing hardware components."""
    instance = VRAMdManager.__new__(VRAMdManager)
    instance.state = ManagerState.RUNNING
    instance.lifecycle = Mock()
    instance.get_resource_budget = Mock(return_value=object())
    return instance


def test_load_model_gets_fresh_budget_and_delegates(manager):
    budget = manager.get_resource_budget.return_value
    backend = object()
    manager.lifecycle.load_model.return_value = backend

    result = manager.load_model("example.gguf", max_gpu_layers=12)

    manager.get_resource_budget.assert_called_once_with()
    manager.lifecycle.load_model.assert_called_once_with(
        name="example.gguf",
        budget=budget,
        max_gpu_layers=12,
    )
    assert result is backend


def test_load_model_passes_none_gpu_layer_limit(manager):
    manager.load_model("example.gguf")

    manager.lifecycle.load_model.assert_called_once_with(
        name="example.gguf",
        budget=manager.get_resource_budget.return_value,
        max_gpu_layers=None,
    )


@pytest.mark.parametrize(
    "state",
    [
        ManagerState.STOPPED,
        ManagerState.STARTING,
        ManagerState.STOPPING,
        ManagerState.FAILED,
    ],
)
def test_load_model_requires_running_manager(manager, state):
    manager.state = state

    with pytest.raises(RuntimeError, match="Model loading requires a running manager"):
        manager.load_model("example.gguf")

    manager.get_resource_budget.assert_not_called()
    manager.lifecycle.load_model.assert_not_called()


def test_unload_model_delegates(manager):
    manager.unload_model("example.gguf")

    manager.lifecycle.unload_model.assert_called_once_with("example.gguf")


@pytest.mark.parametrize(
    "state",
    [
        ManagerState.STOPPED,
        ManagerState.STARTING,
        ManagerState.STOPPING,
        ManagerState.FAILED,
    ],
)
def test_unload_model_requires_running_manager(manager, state):
    manager.state = state

    with pytest.raises(RuntimeError, match="Model unloading requires a running manager"):
        manager.unload_model("example.gguf")

    manager.lifecycle.unload_model.assert_not_called()
