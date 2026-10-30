"""Error paths in verl/ (audit items A7, B16, B19, C22, E34, E35, F40)."""
import importlib
from typing import Any

import pytest
import ray.util
import torch
from omegaconf import OmegaConf

from verl.single_controller.base import worker_group as worker_group_module
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.base.worker import WorkerHelper, WorkerMeta
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.base import _bind_workers_method_to_parent
from verl.utils import distributed
from verl.utils import torch_functional as verl_F
from verl.utils.model import get_model_size
from verl.utils.tracking import Tracking

FLASH_ATTN_MESSAGE = r"flash-attn is required for .*CUDA-only.*uv sync --extra gpu"
CUDA_MESSAGE = r"requires a CUDA GPU"


# --- A7 ---------------------------------------------------------------------------------------


def test_get_model_size_rejects_unknown_scale():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(NotImplementedError, match=r"^Unknown scale X$"):
        get_model_size(model, scale="X")


# --- B16 --------------------------------------------------------------------------------------


class _NeverAliveGroup(WorkerGroup):

    def __init__(self):
        super().__init__(resource_pool=None)
        self._workers = ["w0", "w1"]

    def _is_worker_alive(self, worker) -> bool:
        return False


class _AliveGroup(_NeverAliveGroup):

    def _is_worker_alive(self, worker) -> bool:
        return True


@pytest.fixture
def counted_sleep(monkeypatch):
    """Replace time.sleep in worker_group; abort if the wait loop exceeds any sane bound."""
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) > 100_000:
            raise RuntimeError("wait loop is unbounded")

    monkeypatch.setattr(worker_group_module.time, "sleep", fake_sleep)
    return calls


def test_block_until_all_workers_alive_times_out(counted_sleep):
    group = _NeverAliveGroup()

    with pytest.raises(TimeoutError, match=r"not all 2 workers became alive within \d+s"):
        group._block_until_all_workers_alive()


def test_block_until_all_workers_alive_keeps_one_second_cadence(counted_sleep):
    group = _NeverAliveGroup()

    with pytest.raises(TimeoutError, match="became alive"):
        group._block_until_all_workers_alive(max_wait_seconds=3)

    assert counted_sleep == [1, 1, 1]


def test_block_until_all_workers_alive_returns_once_alive(counted_sleep):
    group = _AliveGroup()

    group._block_until_all_workers_alive()

    assert counted_sleep == []


# --- B19 --------------------------------------------------------------------------------------


class _RaisesAttributeError:

    def __get__(self, instance, owner):
        raise AttributeError("only available on instances")


class _RaisesRuntimeError:

    def __get__(self, instance, owner):
        raise RuntimeError("broken descriptor")


class _UserWorker:
    constant = 3
    instance_only = _RaisesAttributeError()

    @property
    def some_property(self):
        return 1

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def registered(self):
        return "ok"


class _BrokenWorker:
    broken = _RaisesRuntimeError()


class _RecordingGroup(WorkerGroup):
    registered: str  # set at runtime by _bind_worker_method from _func_generator's return value

    def __init__(self):
        super().__init__(resource_pool=None)

    def execute_all(self, *args, **kwargs):
        return "executed"


def _func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    return f"bound:{method_name}"


def test_worker_group_binding_skips_properties_and_attributes():
    group = _RecordingGroup()

    group._bind_worker_method(_UserWorker, _func_generator)

    assert group.registered == "bound:registered"


def test_worker_group_binding_propagates_unexpected_errors():
    group = _RecordingGroup()

    with pytest.raises(RuntimeError, match="broken descriptor"):
        group._bind_worker_method(_BrokenWorker, _func_generator)


def test_colocated_binding_skips_properties_and_attributes():
    parent: Any = type("Parent", (), {})  # methods are bound onto it at runtime

    _bind_workers_method_to_parent(parent, "actor", _UserWorker)

    assert callable(parent.actor_registered)
    assert not hasattr(parent, "actor_constant")


def test_colocated_binding_propagates_unexpected_errors():
    parent = type("Parent", (), {})

    with pytest.raises(RuntimeError, match="broken descriptor"):
        _bind_workers_method_to_parent(parent, "actor", _BrokenWorker)


# --- C22 --------------------------------------------------------------------------------------


def test_node_ip_comes_from_public_ray_api(monkeypatch):
    monkeypatch.setenv("WG_BACKEND", "ray")
    monkeypatch.delenv("MY_HOST_IP", raising=False)
    monkeypatch.delenv("MY_HOST_IPV6", raising=False)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "10.1.2.3")

    assert WorkerHelper()._get_node_ip() == "10.1.2.3"


# --- E34 --------------------------------------------------------------------------------------


def test_require_cuda_passes_when_cuda_present(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert distributed.require_cuda("test component") is None


def test_require_cuda_names_the_component(no_cuda):
    with pytest.raises(RuntimeError, match=r"^test component requires a CUDA GPU"):
        distributed.require_cuda("test component")


def test_initialize_global_process_group_requires_cuda(no_cuda):
    with pytest.raises(RuntimeError, match=CUDA_MESSAGE):
        distributed.initialize_global_process_group()


@pytest.mark.parametrize(
    ("cls_name", "args"),
    [("ActorRolloutRefWorker", (OmegaConf.create({}), "actor")), ("CriticWorker", (OmegaConf.create({}),)),
     ("RewardModelWorker", (OmegaConf.create({}),))],
)
def test_fsdp_worker_requires_cuda(no_cuda, monkeypatch, cls_name, args):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WG_PREFIX", raising=False)
    fsdp_workers = importlib.import_module("verl.workers.fsdp_workers")

    with pytest.raises(RuntimeError, match=CUDA_MESSAGE):
        getattr(fsdp_workers, cls_name)(*args)


# --- E35 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "package"),
    [("verl.workers.actor.dp_actor", "verl.workers.actor"), ("verl.workers.critic.dp_critic", "verl.workers.critic")],
)
def test_dp_module_import_explains_missing_flash_attn(no_flash_attn, fresh_import, module, package):
    with pytest.raises(ImportError, match=FLASH_ATTN_MESSAGE):
        fresh_import(module, package)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (verl_F.log_probs_from_logits_response_rmpad, (None, None, None, 1)),
        (verl_F.log_probs_from_logits_all_rmpad, (None, None, None, 1, 1, 1)),
    ],
)
def test_rmpad_logprob_helpers_explain_missing_flash_attn(no_flash_attn, fn, args):
    with pytest.raises(ImportError, match=FLASH_ATTN_MESSAGE):
        fn(*args)


def test_reward_model_forward_explains_missing_flash_attn(no_flash_attn):
    fsdp_workers = importlib.import_module("verl.workers.fsdp_workers")

    with pytest.raises(ImportError, match=FLASH_ATTN_MESSAGE):
        fsdp_workers.RewardModelWorker._forward_micro_batch(None, None)


# --- F40 --------------------------------------------------------------------------------------


def test_worker_meta_keys_is_immutable():
    assert WorkerMeta.keys == ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                               "CUDA_VISIBLE_DEVICES")


def test_tracking_supported_backends_is_immutable():
    assert Tracking.supported_backend == ("wandb", "mlflow", "console")
