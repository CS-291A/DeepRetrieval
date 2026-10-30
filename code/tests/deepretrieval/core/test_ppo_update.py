"""PPO actor/critic update bookkeeping (audit items G1, D27).

update_policy runs on CPU against a one-parameter fake policy: log_prob == w everywhere, the
reference log-prob is 0, and advantages are 0, so the only live loss term is the KL penalty
(kl_loss_type='kl' gives KL = log_prob - ref_log_prob = w).
"""
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.optim.sgd import (
    SGD,  # torch 2.4 optim has no __all__, so pyright treats torch.optim.SGD as private
)

import verl.utils.torch_functional as verl_F
from verl import DataProto

BATCH = 2
PROMPT_LEN = 3
RESPONSE_LEN = 4
W_INIT = 0.5
# Captured at import so repeated monkeypatching in one test never wraps a wrapper.
_REAL_BACKWARD = torch.Tensor.backward


@pytest.fixture
def dp_actor_module(stub_flash_attn, fresh_import, monkeypatch):
    # update_policy moves each micro-batch to CUDA; keep it on CPU.
    monkeypatch.setattr(TensorDict, "cuda", lambda self, *args, **kwargs: self)
    return fresh_import("verl.workers.actor.dp_actor", "verl.workers.actor")


def _actor_config(kl_loss_coef):
    return OmegaConf.create({
        "ppo_mini_batch_size": BATCH,
        "ppo_micro_batch_size": 1,
        "use_kl_loss": True,
        "kl_loss_coef": kl_loss_coef,
        "kl_loss_type": "kl",
        "clip_ratio": 0.2,
        "entropy_coeff": 0.0,
        "use_dynamic_bsz": False,
        "ulysses_sequence_parallel_size": 1,
        "use_remove_padding": False,
    })


def _rollout_batch():
    seq_len = PROMPT_LEN + RESPONSE_LEN
    batch = TensorDict(
        {
            "responses": torch.ones(BATCH, RESPONSE_LEN, dtype=torch.long),
            "input_ids": torch.ones(BATCH, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(BATCH, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).expand(BATCH, seq_len),
            "old_log_probs": torch.full((BATCH, RESPONSE_LEN), W_INIT),
            "advantages": torch.zeros(BATCH, RESPONSE_LEN),
            "ref_log_prob": torch.zeros(BATCH, RESPONSE_LEN),
        },
        batch_size=[BATCH],
    )
    return DataProto(batch=batch, meta_info={"temperature": 1.0})


def _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef):
    """Run one update_policy call; return (metrics, per-micro-batch losses, w.grad at the optimizer step)."""
    policy = torch.nn.Module()
    policy.w = torch.nn.Parameter(torch.tensor(W_INIT))
    actor = dp_actor_module.DataParallelPPOActor(_actor_config(kl_loss_coef), policy,
                                                 SGD(policy.parameters(), lr=0.0))

    def forward_micro_batch(micro_batch, temperature):
        n = micro_batch["responses"].shape[0]
        return torch.zeros(n, RESPONSE_LEN), policy.w.expand(n, RESPONSE_LEN)

    grads = []

    def optimizer_step():
        grads.append(policy.w.grad.clone())
        return torch.tensor(0.0)

    losses = []

    def recording_backward(self, *args, **kwargs):
        losses.append(self.detach().item())
        return _REAL_BACKWARD(self, *args, **kwargs)

    monkeypatch.setattr(actor, "_forward_micro_batch", forward_micro_batch)
    monkeypatch.setattr(actor, "_optimizer_step", optimizer_step)
    monkeypatch.setattr(torch.Tensor, "backward", recording_backward)
    metrics = actor.update_policy(_rollout_batch())
    return metrics, losses, grads


def test_kl_loss_increases_actor_loss(dp_actor_module, monkeypatch):
    _, losses_without_kl, _ = _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef=0.0)
    _, losses_with_kl, _ = _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef=0.4)

    # loss per micro-batch = kl_coef * KL / grad_accum = 0.4 * 0.5 / 2
    assert losses_without_kl == pytest.approx([0.0, 0.0])
    assert losses_with_kl == pytest.approx([0.1, 0.1])


def test_kl_loss_gradient_pulls_policy_toward_reference(dp_actor_module, monkeypatch):
    _, _, grads = _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef=0.4)

    # d(loss)/dw = +kl_coef summed over the mini-batch: descent lowers w toward ref_log_prob = 0.
    assert [g.item() for g in grads] == pytest.approx([0.4])


def test_kl_loss_metric_keeps_every_micro_batch(dp_actor_module, monkeypatch):
    metrics, _, _ = _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef=0.4)

    assert metrics["actor/kl_loss"] == pytest.approx([W_INIT, W_INIT])
    assert metrics["actor/kl_coef"] == pytest.approx(0.4)


def test_actor_micro_batch_metrics_have_one_entry_per_micro_batch(dp_actor_module, monkeypatch):
    metrics, _, _ = _run_update_policy(dp_actor_module, monkeypatch, kl_loss_coef=0.4)

    lengths = {key: len(metrics[key]) for key in ("actor/entropy_loss", "actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl")}

    assert lengths == dict.fromkeys(lengths, BATCH)
    assert metrics["actor/grad_norm"] == [0.0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_scalars_to_floats_matches_item_bit_for_bit(dtype):
    values = [torch.randn((), dtype=torch.float32).to(dtype) for _ in range(5)]
    mixed = values + [torch.randn((), dtype=torch.float32)]

    result = verl_F.scalars_to_floats(mixed)

    assert result == [v.item() for v in mixed]  # audit: ignore[FLOAT_EQ] bit-identical conversion is the contract
    assert all(type(x) is float for x in result)


def test_scalars_to_floats_rejects_non_scalars():
    with pytest.raises(ValueError, match="0-dim"):
        verl_F.scalars_to_floats([torch.zeros(2)])


@pytest.fixture
def dp_critic_module(stub_flash_attn, fresh_import, monkeypatch):
    monkeypatch.setattr(TensorDict, "cuda", lambda self, *args, **kwargs: self)
    return fresh_import("verl.workers.critic.dp_critic", "verl.workers.critic")


def test_critic_metrics_equal_per_item_reads(dp_critic_module, monkeypatch):
    from verl.trainer.ppo import core_algos

    config = OmegaConf.create({"ppo_mini_batch_size": BATCH, "ppo_micro_batch_size": 1, "use_dynamic_bsz": False,
                               "cliprange_value": 0.5, "model": {}})
    critic_module = torch.nn.Module()
    critic_module.w = torch.nn.Parameter(torch.tensor(0.3))
    critic = dp_critic_module.DataParallelPPOCritic(config, critic_module, SGD(critic_module.parameters(), lr=0.0))
    # bf16 value predictions mixed with fp32 returns: the dtype mix the bf16 critic produces on GPU.
    vpreds = torch.randn(BATCH, RESPONSE_LEN).to(torch.bfloat16)
    monkeypatch.setattr(critic, "_forward_micro_batch",
                        lambda micro_batch: vpreds[micro_batch["row"]].squeeze(0) * critic_module.w.to(torch.bfloat16))
    monkeypatch.setattr(critic, "_optimizer_step", lambda: torch.tensor(0.0))
    seq_len = PROMPT_LEN + RESPONSE_LEN
    returns, values = torch.randn(BATCH, RESPONSE_LEN), torch.randn(BATCH, RESPONSE_LEN)
    batch = TensorDict({"input_ids": torch.ones(BATCH, seq_len, dtype=torch.long),
                        "responses": torch.ones(BATCH, RESPONSE_LEN, dtype=torch.long),
                        "attention_mask": torch.ones(BATCH, seq_len, dtype=torch.long),
                        "position_ids": torch.arange(seq_len).expand(BATCH, seq_len),
                        "values": values, "returns": returns,
                        "row": torch.arange(BATCH).unsqueeze(-1)}, batch_size=[BATCH])
    monkeypatch.setattr(DataProto, "select", lambda self, batch_keys: self)
    eos_mask = torch.ones(1, RESPONSE_LEN, dtype=torch.long)
    expected_losses = [core_algos.compute_value_loss(vpreds=(vpreds[i:i + 1] * torch.tensor(0.3).to(torch.bfloat16)),
                                                     values=values[i:i + 1], returns=returns[i:i + 1], eos_mask=eos_mask,
                                                     cliprange_value=0.5) for i in range(BATCH)]

    metrics = critic.update_critic(DataProto(batch=batch))

    assert metrics["critic/vf_loss"] == [loss.item() for loss, _ in expected_losses]  # audit: ignore[FLOAT_EQ] must be bit-identical to the old per-item reads
