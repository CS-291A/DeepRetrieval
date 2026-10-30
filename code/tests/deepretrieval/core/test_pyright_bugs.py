"""Runtime bugs in verl/ surfaced by pyright (see audit_pyright_verl notes)."""
import runpy
import sys
import types
from typing import cast

import pytest
import torch

from tests.deepretrieval.rewards.conftest import _STUB_MODULES
from verl.single_controller.ray import base as ray_base
from verl.single_controller.ray.base import RayWorkerGroup


class _RemoteMethod:

    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def remote(self, *args, **kwargs):
        self.owner.calls.append((self.name, args, kwargs))
        return f'{self.owner.rank}:{self.name}'


class _FakeWorker:

    def __init__(self, rank):
        self.rank = rank
        self.calls = []

    def __getattr__(self, name):
        return _RemoteMethod(self, name)


def test_execute_rank_zero_sync_calls_only_rank_zero(monkeypatch: pytest.MonkeyPatch):
    workers = [_FakeWorker(0), _FakeWorker(1)]
    group = object.__new__(RayWorkerGroup)
    monkeypatch.setattr(group, '_workers', workers, raising=False)  # fakes stand in for ray ActorHandles
    monkeypatch.setattr(ray_base.ray, 'get', lambda ref: ref)

    result = group.execute_rank_zero_sync('ping', 1, key='v')

    assert result == '0:ping'
    assert workers[0].calls == [('ping', (1,), {'key': 'v'})]
    assert workers[1].calls == []


def test_qwen2_flash_attn_forward_with_output_attentions_returns_no_weights(monkeypatch: pytest.MonkeyPatch):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

    from verl.models.transformers import qwen2 as verl_qwen2

    def fake_flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, **kwargs):
        return query_states  # (bsz, seq, n_heads, head_dim), the layout flash-attn returns

    monkeypatch.setattr(verl_qwen2, '_flash_attention_forward', fake_flash_attention_forward)
    config = Qwen2Config(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, intermediate_size=32,
                         num_hidden_layers=1, vocab_size=32)
    attn = Qwen2Attention(config, layer_idx=0).eval()
    monkeypatch.setattr(attn, '_flash_attn_uses_top_left_mask', False, raising=False)  # set by Qwen2FlashAttention2
    hidden_states = torch.randn(1, 3, 16)
    position_ids = cast(torch.LongTensor, torch.arange(3).unsqueeze(0))  # arange of ints is int64
    position_embeddings = Qwen2RotaryEmbedding(config=config)(hidden_states, position_ids)

    attn_output, attn_weights, _ = verl_qwen2.qwen2_flash_attn_forward(attn,
                                                                       hidden_states,
                                                                       position_ids=position_ids,
                                                                       output_attentions=True,
                                                                       position_embeddings=position_embeddings)

    assert attn_output.shape == (1, 3, 16)
    assert attn_weights is None


REWARD_DEMO_MODULES = [
    "verl.utils.reward_score.fever",
    "verl.utils.reward_score.hotpotqa",
    "verl.utils.reward_score.msmarco_beir",
    "verl.utils.reward_score.nfcorpus",
    "verl.utils.reward_score.scifact",
    "verl.utils.reward_score_dense.fever",
    "verl.utils.reward_score_dense.hotpotqa",
    "verl.utils.reward_score_dense.msmarco_beir",
    "verl.utils.reward_score_dense.nfcorpus",
    "verl.utils.reward_score.msmarco",
    "verl.utils.reward_score.bird",
    "verl.utils.reward_score.bird_no_reason",
    "verl.utils.reward_score.spider",
    "verl.utils.reward_score.spider_no_reason",
]


@pytest.fixture
def stubbed_retrieval_backends(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """pyserini / index stubs from the rewards conftest, run from an empty cwd (no index on disk)."""
    for stub_name, attributes in _STUB_MODULES.items():
        monkeypatch.setitem(sys.modules, stub_name, types.SimpleNamespace(__name__=stub_name, **attributes))
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("module_name", REWARD_DEMO_MODULES)
def test_reward_module_demo_reaches_scoring(module_name, stubbed_retrieval_backends, monkeypatch: pytest.MonkeyPatch):
    """The `python -m <reward module>` demo must call compute_score with its required arguments.

    The demo answer has no <think> block, so it is scored with the -2 format penalty and needs no index.
    """
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    demo_globals = runpy.run_module(module_name, run_name="__main__")

    assert demo_globals["scores"] == -2


def _cpu_memory_buffer(numel, dtype):
    from verl.utils.memory_buffer import MemoryBuffer

    buffer = object.__new__(MemoryBuffer)  # MemoryBuffer.__init__ allocates on CUDA
    buffer.numel = numel
    buffer.numel_padded = numel
    buffer.dtype = dtype
    buffer.data = torch.zeros(numel, dtype=dtype)
    return buffer


def test_build_memory_reference_pads_each_param_with_its_own_dtype():
    from verl.utils.memory_buffer import build_memory_reference_from_module, calc_padded_numel

    module = torch.nn.Module()
    module.a = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    module.b = torch.nn.Parameter(torch.full((3,), 2.0, dtype=torch.float32))
    module.c = torch.nn.Parameter(torch.ones(3, dtype=torch.float16))
    fp32_numel = 2 * calc_padded_numel(torch.Size([3]), torch.float32)
    fp16_numel = calc_padded_numel(torch.Size([3]), torch.float16)
    memory_buffers = {
        torch.float32: _cpu_memory_buffer(fp32_numel, torch.float32),
        torch.float16: _cpu_memory_buffer(fp16_numel, torch.float16),
    }

    build_memory_reference_from_module(module, memory_buffers, maintain_weight=True)

    assert module.b.data.data_ptr() == memory_buffers[torch.float32].data[4:].data_ptr()
    assert torch.equal(module.b.data, torch.full((3,), 2.0))


def test_pad_packed_inputs_rejects_3d_tokens_with_not_implemented():
    from verl.utils.model import pad_packed_inputs

    unpad_tokens = torch.zeros(3, 2, 2)
    cu_seqlens = torch.tensor([0, 3])

    with pytest.raises(NotImplementedError, match="Padding dim 3 is not supported"):
        pad_packed_inputs(unpad_tokens, cu_seqlens, max_seqlen_in_batch=3, size=4)


@pytest.mark.parametrize(("precision", "expected"), [("16", True), ("bf16", True), ("mixed", True), ("8", False)])
def test_precision_type_supported_type(precision, expected):
    from verl.utils.torch_dtypes import PrecisionType

    assert PrecisionType.supported_type(precision) is expected


def test_precision_type_supported_types_lists_every_precision():
    from verl.utils.torch_dtypes import PrecisionType

    assert PrecisionType.supported_types() == ["16", "32", "64", "bf16", "mixed"]


def test_build_rollout_rejects_unknown_rollout_name(monkeypatch: pytest.MonkeyPatch):
    from torch.distributed import device_mesh
    from omegaconf import OmegaConf

    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    monkeypatch.setattr(device_mesh, 'init_device_mesh', lambda *args, **kwargs: None)
    worker = object.__new__(ActorRolloutRefWorker)  # skip __init__: it needs CUDA and a process group
    worker._world_size = 1
    worker.config = OmegaConf.create({'rollout': {'name': 'sglang', 'tensor_model_parallel_size': 1}})

    with pytest.raises(NotImplementedError, match="sglang"):
        worker._build_rollout()


class _PassThroughShardingManager:

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def preprocess_data(self, data):
        return data

    def postprocess_data(self, data):
        return data


def test_compute_rm_score_without_chat_template_switch_scores_the_input_batch(monkeypatch: pytest.MonkeyPatch):
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    from verl import DataProto
    from verl.workers.fsdp_workers import RewardModelWorker

    monkeypatch.setattr(DataProto, 'to', lambda self, device: self)  # no CUDA here
    monkeypatch.setattr(TensorDict, 'cuda', lambda self: self)
    worker = object.__new__(RewardModelWorker)  # skip __init__: it needs CUDA and a process group
    worker._do_switch_chat_template = False
    worker.config = OmegaConf.create({'use_dynamic_bsz': False, 'micro_batch_size': 2})
    monkeypatch.setattr(worker, 'ulysses_sharding_manager', _PassThroughShardingManager(), raising=False)
    worker._forward_micro_batch = lambda micro_batch: micro_batch['input_ids'][:, 0].float()
    data = DataProto.from_dict({
        'input_ids': torch.tensor([[7, 1, 2], [9, 3, 4]]),
        'attention_mask': torch.ones(2, 3, dtype=torch.long),
        'position_ids': torch.arange(3).repeat(2, 1),
        'responses': torch.tensor([[2], [4]]),
    })

    output = worker.compute_rm_score(data)

    assert torch.equal(output.batch['rm_scores'], torch.tensor([[7.0], [9.0]]))
