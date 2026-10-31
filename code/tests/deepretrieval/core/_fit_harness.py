"""Run `RayPPOTrainer.fit` / `_validate` on CPU against fake worker groups.

`fit` is driven through one PPO step with no critic, reference policy or reward model; the
advantage, data and timing metrics are stubbed (they have their own tests) so the step reaches
the reward-metric selection and the tracker. The trainer is a SimpleNamespace standing in for
`self`, as in test_reward_metrics.py.
"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import ray_trainer

RESPONSE_LEN = 3


def token_level_scores(rewards):
    """Rewards placed on the last response token, as RewardManager does."""
    scores = torch.zeros(len(rewards), RESPONSE_LEN)
    scores[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    return scores


class EchoRollout:
    world_size = 1

    def generate_sequences(self, batch):
        n = len(batch)
        return DataProto.from_dict(tensors={"responses": torch.ones(n, RESPONSE_LEN, dtype=torch.long),
                                            "attention_mask": torch.ones(n, 2 + RESPONSE_LEN, dtype=torch.long)})


def _prompt_batch(n, data_sources=None):
    batch: dict[str, torch.Tensor | np.ndarray] = {
        "input_ids": torch.ones(n, 2, dtype=torch.long),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "position_ids": torch.arange(2).expand(n, 2),
    }
    if data_sources is not None:
        batch["data_source"] = np.array(data_sources, dtype=object)
    return batch


def run_validate(data_sources, rewards):
    trainer: Any = SimpleNamespace(
        val_dataloader=[_prompt_batch(len(rewards), data_sources)],
        config=OmegaConf.create({"reward_model": {"enable": False}}),
        tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=0),
        actor_rollout_wg=EchoRollout(),
        val_reward_fn=lambda batch: token_level_scores(rewards),
    )
    return ray_trainer.RayPPOTrainer._validate(trainer)


def fit_config(train_files, logger="console", val_only=False):
    return OmegaConf.create({
        "data": {"train_files": train_files},
        "trainer": {"project_name": "p", "experiment_name": "e", "logger": logger, "total_epochs": 1,
                    "critic_warmup": 10**9, "test_freq": 0, "save_freq": 0, "val_before_train": val_only,
                    "val_only": val_only},
        "actor_rollout_ref": {"rollout": {"n": 1}, "actor": {"use_kl_loss": True}},
        "algorithm": {"adv_estimator": "gae", "gamma": 1.0, "lam": 1.0, "kl_penalty": "kl"},
        "reward_model": {"enable": False},
    })


def stub_step_metrics(monkeypatch):
    """Skip advantage/data/timing computations, which need critic outputs the fake step lacks."""
    monkeypatch.setattr(ray_trainer, "compute_advantage", lambda batch, **kwargs: batch)
    monkeypatch.setattr(ray_trainer, "compute_data_metrics", lambda batch, use_critic: {})
    monkeypatch.setattr(ray_trainer, "compute_timing_metrics", lambda batch, timing_raw: {})


def make_fit_trainer(config, rewards, train_dataloader=None, val_reward_fn=None):
    """A duck-typed RayPPOTrainer `self` that runs one training step of len(rewards) samples."""
    trainer: Any = SimpleNamespace(
        config=config,
        train_dataloader=[_prompt_batch(len(rewards))] if train_dataloader is None else train_dataloader,
        val_dataloader=[_prompt_batch(len(rewards), ["nq_serini"] * len(rewards))],
        tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=0),
        actor_rollout_wg=EchoRollout(),
        use_reference_policy=False,
        use_critic=False,
        use_rm=False,
        reward_fn=lambda batch: token_level_scores(rewards),
        val_reward_fn=val_reward_fn,
        kl_ctrl=None,
        total_training_steps=1,
        _balance_batch=lambda batch, metrics: None,
        _save_checkpoint=lambda: None,
    )
    trainer._validate = lambda: ray_trainer.RayPPOTrainer._validate(trainer)
    trainer._fit = lambda logger: ray_trainer.RayPPOTrainer._fit(trainer, logger)
    return trainer


def run_fit(trainer):
    ray_trainer.RayPPOTrainer.fit(trainer)
