"""Characterisation of metric selection by data_source / train_files (audit item 15).

Pins which validation metric `_validate` emits for every routable data_source and which
training reward metric `fit` logs for every `data.train_files` used by scripts/train/.
Written against the substring chains in ray_trainer.py before they were replaced by
verl/trainer/reward_registry.py; the refactor had to keep every case green.

Rewards [6, 3, 1.2, 0.6, 0] give a different result under each metric family:
ndcg (0.1 + ndcg) 2.08, answer-span recall@k, SQL accuracy (> 1) 0.6, "== 3" 0.2.
"""
import pytest
from omegaconf import ListConfig

import verl.utils.tracking
from tests.deepretrieval.core._fit_harness import (
    fit_config,
    make_fit_trainer,
    run_fit,
    run_validate,
    stub_step_metrics,
    token_level_scores,
)
from verl import DataProto
from verl.trainer import reward_registry

REWARDS = [6.0, 3.0, 1.2, 0.6, 0.0]


def _ndcg(ds):
    return {f"val/test_score/{ds}": pytest.approx((5.9 + 2.9 + 1.1 + 0.5) / 5)}


def _recall(ds):
    hits = {5: 0.2, 20: 0.2, 50: 0.4, 100: 0.4, 1000: 0.4, 3000: 0.6}
    return {f"val/recall@{k}/{ds}": pytest.approx(v) for k, v in hits.items()}


def _sql(ds):
    return {f"val/test_score/{ds}": pytest.approx(0.6)}


def _equals_3(ds):
    return {f"val/test_score/{ds}": pytest.approx(0.2)}


def _none(ds):
    return {}


VALIDATION = {
    "screening": _equals_3,
    "pubmed": _none, "pubmed_32": _none, "pmd_no_reason": _none, "ctgov": _none, "ctgov_32": _none,
    "fiqa_test": _ndcg, "scifact_test": _ndcg, "scifact_dense_test": _ndcg, "nfcorpus_test": _ndcg,
    "nfcorpus_dense_test": _ndcg, "hotpotqa_test": _ndcg, "hotpotqa_dense_test": _ndcg, "fever_test": _ndcg,
    "fever_dense_test": _ndcg, "msmarco_beir_test": _ndcg, "msmarco_beir_dense_test": _ndcg,
    "msmarco_health_sparse_test": _ndcg, "msmarco_tech_dense_test": _ndcg,
    "nq_serini": _recall, "nq_serini_no_reason": _recall, "triviaqa": _recall, "triviaqa_no_reason": _recall,
    "squad": _recall, "squad_no_reason": _recall,
    "bird_test": _sql, "bird_wor_test": _sql, "spider_test": _sql, "spider_wor_test": _sql,
    # a source with no reward table (only reachable with a custom val_reward_fn)
    "some_other_task": _equals_3,
}


@pytest.mark.parametrize(("data_source", "expected"), list(VALIDATION.items()))
def test_validation_metric_for_data_source(data_source, expected):
    metrics = run_validate([data_source] * len(REWARDS), REWARDS)

    assert metrics == expected(data_source)


def test_validation_metrics_are_per_data_source_in_a_mixed_batch():
    metrics = run_validate(["bird_test", "fiqa_test"], [1.1 + 0.1, 0.1 + 0.5])

    assert metrics == {"val/test_score/bird_test": pytest.approx(1.0),
                       "val/test_score/fiqa_test": pytest.approx(0.5)}


TRAINING = {
    "data/local_index_search/fever/dense/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/fever/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/fiqa/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/hotpotqa/dense/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/hotpotqa/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/msmarco_health/sparse/train.parquet": "compute_reward_metrics_recall_ndcg",
    "data/local_index_search/msmarco_tech/dense/train.parquet": "compute_reward_metrics_recall_ndcg",
    "data/local_index_search/msmarco_beir/dense/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/msmarco_beir/sparse/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/nfcorpus/dense/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/nfcorpus/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/no_reason/nq_serini_no_reason/train.parquet": "compute_reward_metrics_nq_serini",
    "data/local_index_search/no_reason/squad_no_reason/train.parquet": "compute_reward_metrics_nq_serini",
    "data/local_index_search/no_reason/triviaqa_no_reason/train.parquet": "compute_reward_metrics_nq_serini",
    "data/local_index_search/nq_serini/train.parquet": "compute_reward_metrics_nq_serini",
    "data/local_index_search/scifact/dense/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/scifact/sparse/train.parquet": "compute_reward_metrics_ndcg",
    "data/local_index_search/squad/train.parquet": "compute_reward_metrics_nq_serini",
    "data/local_index_search/triviaqa/train.parquet": "compute_reward_metrics_nq_serini",
    "data/search_engine/ctgov_32/train.parquet": "compute_reward_metrics_search_engine",
    "data/search_engine/ctgov/train.parquet": "compute_reward_metrics_search_engine",
    "data/search_engine/no_reason/SearchEngine/pmd_no_reason/train.parquet": "compute_reward_metrics_search_engine",
    "data/search_engine/pubmed_32/train.parquet": "compute_reward_metrics_search_engine",
    "data/search_engine/pubmed/train.parquet": "compute_reward_metrics_search_engine",
    "data/sql/bird/train.parquet": "compute_reward_metrics_sql",
    "data/sql/spider/train.parquet": "compute_reward_metrics_sql",
    "data/sql/sql_all/train.parquet": "compute_reward_metrics_sql",
    "data/sql/wor/spider_wor/train.parquet": "compute_reward_metrics_sql",
    "data/screening/train.parquet": "compute_reward_metrics_search_engine",
}


class _RecordingTracking:
    instances = []

    def __init__(self, **kwargs):
        self.logged = []
        _RecordingTracking.instances.append(self)

    def log(self, data, step, backend=None):
        self.logged.append((step, data))

    def finish(self, exit_code):
        pass


def _logged_training_metrics(monkeypatch, train_files):
    monkeypatch.setattr(_RecordingTracking, "instances", [])
    monkeypatch.setattr(verl.utils.tracking, "Tracking", _RecordingTracking)
    stub_step_metrics(monkeypatch)
    run_fit(make_fit_trainer(fit_config(train_files), REWARDS))
    [tracker] = _RecordingTracking.instances
    [(step, metrics)] = tracker.logged
    assert step == 1
    return metrics


def _expected_training_metrics(function_name):
    batch = DataProto.from_dict(tensors={"token_level_scores": token_level_scores(REWARDS)})
    return getattr(reward_registry, function_name)(batch)


@pytest.mark.parametrize(("train_files", "function_name"), list(TRAINING.items()))
def test_training_reward_metric_for_train_files(monkeypatch, train_files, function_name):
    metrics = _logged_training_metrics(monkeypatch, train_files)

    assert metrics == pytest.approx(_expected_training_metrics(function_name))


def test_training_reward_metric_for_a_list_of_train_files_falls_back_to_search_engine(monkeypatch):
    # `in` on a ListConfig tests membership, not substrings: no list of paths matches any marker.
    metrics = _logged_training_metrics(monkeypatch, ListConfig(["data/sql/bird/train.parquet"]))

    assert metrics == pytest.approx(_expected_training_metrics("compute_reward_metrics_search_engine"))
