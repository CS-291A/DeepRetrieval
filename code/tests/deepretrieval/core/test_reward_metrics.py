"""Training and validation reward metrics invert the reward step tables (audit items G3, D28).

Reward tables (read from verl/utils/reward_score/, total = format reward + answer score):
- nq_serini / triviaqa / squad and *_no_reason: 1 + {5, 4, 2, 1, 0.5, 0.1} for answer rank
  <= {5, 20, 50, 100, 1000, 3000}; miss 1 - 3.5 (squad: 1 - 2.5); search error 1 - 4 (squad: 1 - 3);
  bad format -4 + 0.
- scifact / fiqa / nfcorpus / hotpotqa / fever / msmarco(_beir): 0.1 + ndcg if ndcg > 0, else 0; bad format -2.
- bird / spider: 0.1 + 1 if execution results match; bird wrong result 0.1 + 0.3; otherwise 0;
  bad format -2.
"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer import reward_registry
from verl.trainer.ppo import ray_trainer

RESPONSE_LEN = 3

# One sample per outcome of the answer-span reward table, best first.
ANSWER_SPAN_REWARDS = [
    1 + 5,  # rank <= 5
    1 + 4,  # rank <= 20
    1 + 2,  # rank <= 50
    1 + 1,  # rank <= 100
    1 + 0.5,  # rank <= 1000
    1 + 0.1,  # rank <= 3000
    1 - 3.5,  # nq/triviaqa miss
    1 - 2.5,  # squad miss
    1 - 4,  # nq/triviaqa search error
    -4 + 0,  # bad format
]


def _token_level_scores(rewards):
    """Rewards placed on the last response token, as RewardManager does."""
    scores = torch.zeros(len(rewards), RESPONSE_LEN)
    scores[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    return scores


def _training_batch(rewards):
    return DataProto.from_dict(tensors={"token_level_scores": _token_level_scores(rewards)})


def test_training_hit_at_k_matches_reward_table():
    metrics = reward_registry.compute_reward_metrics_nq_serini(_training_batch(ANSWER_SPAN_REWARDS))

    hits = {k: metrics[f"reward/hit@{k}"] for k in (5, 20, 50, 100, 1000, 3000)}

    assert hits == pytest.approx({5: 0.1, 20: 0.2, 50: 0.3, 100: 0.4, 1000: 0.5, 3000: 0.6})


def test_training_wrong_answer_excludes_format_errors():
    metrics = reward_registry.compute_reward_metrics_nq_serini(_training_batch(ANSWER_SPAN_REWARDS))

    assert metrics["reward/wrong_answer_ratio"] == pytest.approx(0.3)
    assert metrics["reward/format_error_ratio"] == pytest.approx(0.1)


def test_training_ndcg_metric_for_msmarco_matches_reward_table():
    rewards = [0.1 + 0.5, 0.1 + 0.25, 0, -2]

    metrics = reward_registry.compute_reward_metrics_recall_ndcg(_training_batch(rewards))

    assert metrics["reward/ndcg"] == pytest.approx(0.75 / 4)


def test_training_sql_metrics_match_reward_table():
    rewards = [0.1 + 1, 0.1 + 0.3, 0, -2]

    metrics = reward_registry.compute_reward_metrics_sql(_training_batch(rewards))

    assert metrics["reward/accuracy"] == pytest.approx(0.25)
    assert metrics["reward/wrong_answer_ratio"] == pytest.approx(0.5)
    assert metrics["reward/format_error_ratio"] == pytest.approx(0.25)


class _EchoRollout:
    world_size = 1

    def generate_sequences(self, batch):
        return DataProto.from_dict(tensors={"responses": torch.ones(len(batch), RESPONSE_LEN, dtype=torch.long)})


def _validate(data_sources, rewards):
    n = len(rewards)
    test_data = {
        "input_ids": torch.ones(n, 2, dtype=torch.long),
        "attention_mask": torch.ones(n, 2, dtype=torch.long),
        "position_ids": torch.arange(2).expand(n, 2),
        "data_source": np.array(data_sources, dtype=object),
    }
    trainer: Any = SimpleNamespace(  # duck-typed stand-in for the RayPPOTrainer `self`
        val_dataloader=[test_data],
        config=OmegaConf.create({"reward_model": {"enable": False}}),
        tokenizer=SimpleNamespace(eos_token_id=0, pad_token_id=0),
        actor_rollout_wg=_EchoRollout(),
        val_reward_fn=lambda batch: _token_level_scores(rewards),
    )
    return ray_trainer.RayPPOTrainer._validate(trainer)


def test_validation_recall_at_k_matches_training_hit_at_k():
    metrics = _validate(["nq_serini_test"] * len(ANSWER_SPAN_REWARDS), ANSWER_SPAN_REWARDS)

    recalls = {k: metrics[f"val/recall@{k}/nq_serini_test"] for k in (5, 20, 50, 100, 1000, 3000)}

    assert recalls == pytest.approx({5: 0.1, 20: 0.2, 50: 0.3, 100: 0.4, 1000: 0.5, 3000: 0.6})


def test_validation_recall_keys_are_per_data_source():
    metrics = _validate(["nq_serini_test", "squad_test"], [1 + 5, 1 - 2.5])

    assert metrics == pytest.approx({
        **{f"val/recall@{k}/nq_serini_test": 1.0 for k in (5, 20, 50, 100, 1000, 3000)},
        **{f"val/recall@{k}/squad_test": 0.0 for k in (5, 20, 50, 100, 1000, 3000)},
    })


def test_validation_sql_accuracy_uses_shared_threshold():
    metrics = _validate(["bird_test"] * 4, [0.1 + 1, 0.1 + 0.3, 0, -2])

    assert metrics == pytest.approx({"val/test_score/bird_test": 0.25})


def test_validation_ndcg_uses_shared_formula():
    metrics = _validate(["msmarco_test"] * 4, [0.1 + 0.5, 0.1 + 0.25, 0, -2])

    assert metrics == pytest.approx({"val/test_score/msmarco_test": 0.75 / 4})


REWARD_HELPERS = [reward_registry.compute_answer_span_hit_ratios, reward_registry.compute_mean_ndcg,
                  reward_registry.compute_sql_accuracy]


@pytest.mark.parametrize("helper", REWARD_HELPERS, ids=lambda helper: helper.__name__)
def test_reward_helper_rejects_a_2d_tensor(helper):
    # token-level scores must be summed to one reward per sample first
    with pytest.raises(ValueError, match=r"^expected a non-empty 1-D reward tensor, got shape \(2, 1\)$"):
        helper(torch.tensor([[6.0], [1.1]]))


@pytest.mark.parametrize("helper", REWARD_HELPERS, ids=lambda helper: helper.__name__)
def test_reward_helper_rejects_an_empty_tensor(helper):
    with pytest.raises(ValueError, match=r"^expected a non-empty 1-D reward tensor, got shape \(0,\)$"):
        helper(torch.tensor([]))


def test_sql_reward_of_exactly_the_threshold_is_not_accurate():
    rewards = torch.tensor([reward_registry.SQL_ACCURATE_MIN_REWARD])

    assert reward_registry.compute_sql_accuracy(rewards) == pytest.approx(0.0)


def test_sql_reward_just_above_the_threshold_is_accurate():
    threshold = torch.tensor([reward_registry.SQL_ACCURATE_MIN_REWARD])

    rewards = torch.nextafter(threshold, torch.tensor([2.0]))

    assert reward_registry.compute_sql_accuracy(rewards) == pytest.approx(1.0)
