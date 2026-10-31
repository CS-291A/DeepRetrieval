# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward routing: the one table that says how each data_source is scored and measured.

`ROUTES` is searched in order for the first row whose `marker` is a substring of the
data_source (order matters: 'msmarco_beir' before 'msmarco'). The row gives
- the reward module (imported lazily; `variant` picks e.g. the dense or no-reason module),
- the extra keyword arguments RewardManager passes to its `compute_score`,
- the validation metric `_validate` reports for that source, and
- the training reward metric `fit` logs when `data.train_files` matches the marker.

Adding a dataset is one row here. The metric functions invert the reward step tables in
verl/utils/reward_score/, so changing a reward table means checking its metric below.
"""
import importlib
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from verl.utils.paths import require_existing, resolve_data_reference
from verl.utils.torch_functional import scalars_to_floats

# Reward-table inversions shared by the training and validation metrics.
# Keep them in sync with the step tables in verl/utils/reward_score/.

# nq_serini / triviaqa / squad (+ *_no_reason): total reward = format 1 + answer score 5/4/2/1/0.5/0.1 for an
# answer-span rank <= 5/20/50/100/1000/3000, so "hit within top-k" <=> reward >= the band value for k.
# Misses and search errors are 1 - 2.5 .. 1 - 4 (all < 1.1); a bad format is -4 + 0.
ANSWER_SPAN_HIT_MIN_REWARD = ((5, 6.0), (20, 5.0), (50, 3.0), (100, 2.0), (1000, 1.5), (3000, 1.1))
ANSWER_SPAN_FORMAT_ERROR_REWARD = -4
# BEIR / msmarco ndcg tasks: total reward = 0.1 + ndcg when ndcg > 0, else 0 (or -2 for a bad format).
NDCG_FORMAT_REWARD = 0.1
# bird / spider: 0.1 + 1 for matching execution results; every other outcome is <= 0.1 + 0.3.
SQL_ACCURATE_MIN_REWARD = 1.0


def compute_answer_span_hit_ratios(reward_tensor: torch.Tensor) -> dict[int, float]:
    """Fraction of samples whose answer span is retrieved within the top-k, for each k of the reward table."""
    if reward_tensor.dim() != 1 or reward_tensor.numel() == 0:
        raise ValueError(f'expected a non-empty 1-D reward tensor, got shape {tuple(reward_tensor.shape)}')
    ratios = [(reward_tensor >= min_reward).float().mean() for _, min_reward in ANSWER_SPAN_HIT_MIN_REWARD]
    return dict(zip([k for k, _ in ANSWER_SPAN_HIT_MIN_REWARD], scalars_to_floats(ratios), strict=True))


def compute_mean_ndcg(reward_tensor: torch.Tensor) -> float:
    """Mean ndcg over all samples (samples without a positive ndcg count as 0)."""
    if reward_tensor.dim() != 1 or reward_tensor.numel() == 0:
        raise ValueError(f'expected a non-empty 1-D reward tensor, got shape {tuple(reward_tensor.shape)}')
    positive = reward_tensor[reward_tensor > NDCG_FORMAT_REWARD]
    return (torch.sum(positive - NDCG_FORMAT_REWARD).float() / reward_tensor.numel()).item()


def compute_sql_accuracy(reward_tensor: torch.Tensor) -> float:
    """Fraction of samples whose SQL execution results match the gold query."""
    if reward_tensor.dim() != 1 or reward_tensor.numel() == 0:
        raise ValueError(f'expected a non-empty 1-D reward tensor, got shape {tuple(reward_tensor.shape)}')
    return (reward_tensor > SQL_ACCURATE_MIN_REWARD).float().mean().item()


def compute_reward_metrics_search_engine(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    # Calculate all_correct ratio (value == 3)
    seventy_recall = torch.sum(reward_tensor == 6).float() / reward_tensor.numel()
    reward_metrics["reward/70_recall_ratio"] = seventy_recall.detach().item()
    
    fifty_recall = torch.sum(reward_tensor >= 5).float() / reward_tensor.numel()
    reward_metrics["reward/50_recall_ratio"] = fifty_recall.detach().item()
    
    forty_recall = torch.sum(reward_tensor >= 4).float() / reward_tensor.numel()
    reward_metrics["reward/40_recall_ratio"] = forty_recall.detach().item()
    
    thirty_recall = torch.sum(reward_tensor >= 2).float() / reward_tensor.numel()
    reward_metrics["reward/30_recall_ratio"] = thirty_recall.detach().item()
    
    ten_recall = torch.sum(reward_tensor >= 1.5).float() / reward_tensor.numel()
    reward_metrics["reward/10_recall_ratio"] = ten_recall.detach().item()
    
    five_recall = torch.sum(reward_tensor >= 1.1).float() / reward_tensor.numel()
    reward_metrics["reward/5_recall_ratio"] = five_recall.detach().item()
    
    
    # Calculate format_error ratio (value == -1)
    format_error = torch.sum(reward_tensor == -4).float() / reward_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
    # Calculate wrong answer ratio (value == -1)
    format_error = torch.sum(reward_tensor == -2.5).float() / reward_tensor.numel()
    reward_metrics["reward/wrong_answer_ratio"] = format_error.detach().item()

    return reward_metrics

def compute_reward_metrics_nq_serini(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    for k, hit_ratio in compute_answer_span_hit_ratios(reward_tensor).items():
        reward_metrics[f"reward/hit@{k}"] = hit_ratio

    format_error = torch.sum(reward_tensor == ANSWER_SPAN_FORMAT_ERROR_REWARD).float() / reward_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()

    # well-formatted answers that miss the top-3000 (or whose search failed)
    lowest_hit_reward = ANSWER_SPAN_HIT_MIN_REWARD[-1][1]
    wrong_answer = (reward_tensor > ANSWER_SPAN_FORMAT_ERROR_REWARD) & (reward_tensor < lowest_hit_reward)
    reward_metrics["reward/wrong_answer_ratio"] = (torch.sum(wrong_answer).float() / reward_tensor.numel()).item()

    return reward_metrics


def compute_reward_metrics_ndcg(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    format_score = NDCG_FORMAT_REWARD

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    # Calculate all_correct ratio (value == 3)
    all_correct = torch.sum(reward_tensor > format_score).float() / reward_tensor.numel()
    reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
    # Calculate format_error ratio (value == -1)
    format_error = torch.sum(reward_tensor < 0).float() / reward_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
    # Calculate wrong answer ratio (value == -1)
    all_wrong = torch.sum(reward_tensor == 0).float() / reward_tensor.numel()
    reward_metrics["reward/wrong_answer_ratio"] = all_wrong.detach().item()
    
    # avg value of reward > format_score
    reward_metrics["reward/ndcg"] = compute_mean_ndcg(reward_tensor)
    
    return reward_metrics

def compute_reward_metrics_recall_ndcg(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    format_score = NDCG_FORMAT_REWARD

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    # Calculate all_correct ratio (value == 3)
    all_correct = torch.sum(reward_tensor > format_score).float() / reward_tensor.numel()
    reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
    # Calculate format_error ratio (value == -1)
    format_error = torch.sum(reward_tensor < 0).float() / reward_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
    # Calculate wrong answer ratio (value == -1)
    all_wrong = torch.sum(reward_tensor == 0).float() / reward_tensor.numel()
    reward_metrics["reward/wrong_answer_ratio"] = all_wrong.detach().item()
    
    # msmarco's reward no longer adds a recall bonus: reward = format_score + ndcg
    reward_metrics["reward/ndcg"] = compute_mean_ndcg(reward_tensor)
    
    return reward_metrics


def compute_reward_metrics_sql(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}

    # accuracy
    accuracy = compute_sql_accuracy(reward_tensor)
    reward_metrics["reward/accuracy"] = accuracy

    # normal
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    # Calculate all_correct ratio (value == 1.1)
    reward_metrics["reward/all_correct_ratio"] = accuracy
    # Calculate format_error ratio (value == -2)
    format_error = torch.sum(reward_tensor < 0).float() / reward_tensor.numel()
    reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
    # Calculate wrong answer ratio: well formatted but not accurate (0, or 0.1 + 0.3 for bird)
    all_wrong = torch.sum((reward_tensor >= 0) & (reward_tensor <= SQL_ACCURATE_MIN_REWARD)).float() / reward_tensor.numel()
    reward_metrics["reward/wrong_answer_ratio"] = all_wrong.detach().item()
    

    return reward_metrics


# ---------------------------------------------------------------------------------------------
# Validation metrics: (data_source, per-sample total rewards) -> {metric key: value}


def no_validation_metric(data_source: str, rewards: list[float]) -> dict[str, float]:
    """pubmed / ctgov: no validation metric is reported."""
    return {}


def validation_mean_ndcg(data_source: str, rewards: list[float]) -> dict[str, float]:
    return {f'val/test_score/{data_source}': compute_mean_ndcg(torch.tensor(rewards))}


def validation_answer_span_recall(data_source: str, rewards: list[float]) -> dict[str, float]:
    hit_ratios = compute_answer_span_hit_ratios(torch.tensor(rewards))
    return {f'val/recall@{k}/{data_source}': hit_ratio for k, hit_ratio in hit_ratios.items()}


def validation_sql_accuracy(data_source: str, rewards: list[float]) -> dict[str, float]:
    return {f'val/test_score/{data_source}': compute_sql_accuracy(torch.tensor(rewards))}


def validation_reward_equals_3(data_source: str, rewards: list[float]) -> dict[str, float]:
    """Sources without a reward table to invert (screening, or any unrouted source)."""
    count_equal_3 = sum(1 for reward in rewards if reward == 3)
    total_count = len(rewards)
    return {f'val/test_score/{data_source}': count_equal_3 / total_count if total_count > 0 else 0}


# ---------------------------------------------------------------------------------------------
# Extra compute_score keyword arguments: (non_tensor_batch of one sample, data_source, APIs) -> kwargs


@dataclass(frozen=True)
class SearchApis:
    """The search-engine clients RewardManager owns."""
    pubmed: Any
    ctgov: Any


def no_extra_kwargs(sample: Mapping[str, Any], data_source: str, apis: SearchApis) -> dict[str, Any]:
    return {}


def publication_search_kwargs(sample: Mapping[str, Any], data_source: str, apis: SearchApis) -> dict[str, Any]:
    return {'search_api': apis.pubmed, 'literature_type': 'publication', 'pub_date': sample['pub_date']}


def trial_search_kwargs(sample: Mapping[str, Any], data_source: str, apis: SearchApis) -> dict[str, Any]:
    return {'search_api': apis.ctgov, 'literature_type': 'trial', 'pub_date': sample['pub_date']}


def data_source_kwargs(sample: Mapping[str, Any], data_source: str, apis: SearchApis) -> dict[str, Any]:
    return {'data_source': data_source}


def sql_kwargs(sample: Mapping[str, Any], data_source: str, apis: SearchApis) -> dict[str, Any]:
    # Parquets store `data/...` (relative to code/); re-root it under the data dir. sqlite3.connect
    # would silently create an empty database for a missing file, so a missing one fails here.
    db_path = require_existing(resolve_data_reference(sample['extra_info']['db_path']), f'{data_source} SQLite database')
    return {'data_source': data_source, 'db_path': str(db_path)}


# ---------------------------------------------------------------------------------------------
# The table

KwargsBuilder = Callable[[Mapping[str, Any], str, SearchApis], dict[str, Any]]
ValidationMetric = Callable[[str, list[float]], dict[str, float]]
TrainingMetric = Callable[[Any], dict[str, Any]]  # DataProto with batch['token_level_scores'] -> metrics


@dataclass(frozen=True)
class RewardRoute:
    marker: str  # substring of data_source (and of data.train_files) that selects this row
    module: str  # module with compute_score
    build_kwargs: KwargsBuilder
    validation_metric: ValidationMetric
    training_metric: TrainingMetric
    variant: tuple[str, str] | None = None  # (second marker, module used instead when it is also present)

    def module_for(self, data_source: str) -> str:
        if self.variant is not None and self.variant[0] in data_source:
            return self.variant[1]
        return self.module

    def compute_score_fn(self, data_source: str) -> Callable[..., Any]:
        return importlib.import_module(self.module_for(data_source)).compute_score


_SPARSE = 'verl.utils.reward_score.'
_DENSE = 'verl.utils.reward_score_dense.'

# Short names for the table columns.
_ndcg_metrics = (validation_mean_ndcg, compute_reward_metrics_ndcg)
_search_metrics = (no_validation_metric, compute_reward_metrics_search_engine)
_answer_span_metrics = (validation_answer_span_recall, compute_reward_metrics_nq_serini)
_sql_metrics = (validation_sql_accuracy, compute_reward_metrics_sql)

ROUTES: tuple[RewardRoute, ...] = (
    # marker, reward module, kwargs, (validation metric, training metric), variant
    RewardRoute('screening', _SPARSE + 'screening', no_extra_kwargs,
                validation_reward_equals_3, compute_reward_metrics_search_engine),
    RewardRoute('pubmed', _SPARSE + 'pubmed', publication_search_kwargs, *_search_metrics),
    RewardRoute('pmd_no_reason', _SPARSE + 'pmd_no_reason', publication_search_kwargs, *_search_metrics),
    RewardRoute('ctgov', _SPARSE + 'ctgov', trial_search_kwargs, *_search_metrics),
    RewardRoute('fiqa', _SPARSE + 'fiqa', data_source_kwargs, *_ndcg_metrics),  # no dense fiqa module exists
    RewardRoute('scifact', _SPARSE + 'scifact', data_source_kwargs, *_ndcg_metrics, ('dense', _DENSE + 'scifact')),
    RewardRoute('nfcorpus', _SPARSE + 'nfcorpus', data_source_kwargs, *_ndcg_metrics, ('dense', _DENSE + 'nfcorpus')),
    RewardRoute('nq_serini', _SPARSE + 'nq_serini', no_extra_kwargs, *_answer_span_metrics,
                ('no_reason', _SPARSE + 'nq_no_reason')),
    RewardRoute('triviaqa', _SPARSE + 'triviaqa', no_extra_kwargs, *_answer_span_metrics,
                ('no_reason', _SPARSE + 'tqa_no_reason')),
    RewardRoute('squad', _SPARSE + 'squad', no_extra_kwargs, *_answer_span_metrics,
                ('no_reason', _SPARSE + 'squad_no_reason')),
    RewardRoute('hotpotqa', _SPARSE + 'hotpotqa', data_source_kwargs, *_ndcg_metrics, ('dense', _DENSE + 'hotpotqa')),
    RewardRoute('fever', _SPARSE + 'fever', data_source_kwargs, *_ndcg_metrics, ('dense', _DENSE + 'fever')),
    RewardRoute('msmarco_beir', _SPARSE + 'msmarco_beir', data_source_kwargs, *_ndcg_metrics,
                ('dense', _DENSE + 'msmarco_beir')),
    # msmarco subsets: one module, which reads sparse/dense from the data_source itself
    RewardRoute('msmarco', _SPARSE + 'msmarco', data_source_kwargs,
                validation_mean_ndcg, compute_reward_metrics_recall_ndcg),
    RewardRoute('bird', _SPARSE + 'bird', sql_kwargs, *_sql_metrics, ('wor', _SPARSE + 'bird_no_reason')),
    RewardRoute('spider', _SPARSE + 'spider', sql_kwargs, *_sql_metrics, ('wor', _SPARSE + 'spider_no_reason')),
)

# data/sql/sql_all mixes bird and spider, so its train_files path names neither.
SQL_TRAIN_FILES_MARKER = 'sql'


def find_route(key: Container[str]) -> RewardRoute | None:
    """First row whose marker is `in` key (a substring of a str; an element of a list of paths)."""
    return next((route for route in ROUTES if route.marker in key), None)


def require_route(data_source: str) -> RewardRoute:
    route = find_route(data_source)
    if route is None:
        raise NotImplementedError(f"No reward function for data_source '{data_source}'")
    return route


def select_compute_score(data_source: str) -> Callable[..., Any]:
    return require_route(data_source).compute_score_fn(data_source)


def validation_metrics(data_source: str, rewards: list[float]) -> dict[str, float]:
    route = find_route(data_source)
    metric = validation_reward_equals_3 if route is None else route.validation_metric
    return metric(data_source, rewards)


def select_training_metrics(train_files: Container[str]) -> TrainingMetric:
    route = find_route(train_files)
    if route is not None:
        return route.training_metric
    if SQL_TRAIN_FILES_MARKER in train_files:
        return compute_reward_metrics_sql
    return compute_reward_metrics_search_engine
