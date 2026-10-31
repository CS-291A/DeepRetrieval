"""The reward registry is the one table that routes a data_source (audit item 15).

Behaviour (which module, kwargs and metrics each source gets) is pinned by the characterisation
tests in rewards/test_reward_routing.py and core/test_reward_metric_routing.py; these tests pin
properties of the table itself.
"""
import importlib.util

import pytest

from verl.trainer import reward_registry


@pytest.mark.parametrize("route", reward_registry.ROUTES, ids=lambda route: route.marker)
def test_every_route_names_an_existing_reward_module(route):
    modules = [route.module] + ([route.variant[1]] if route.variant else [])

    missing = [module for module in modules if importlib.util.find_spec(module) is None]

    assert missing == []


def test_route_markers_are_unique():
    markers = [route.marker for route in reward_registry.ROUTES]

    assert len(markers) == len(set(markers))


def test_find_route_returns_none_for_a_source_without_a_reward():
    assert reward_registry.find_route("wikisql_train") is None


def test_find_route_honours_table_order_for_overlapping_markers():
    # 'msmarco' is a substring of 'msmarco_beir'; the more specific row must come first.
    route = reward_registry.find_route("msmarco_beir_test")

    assert route is not None and route.marker == "msmarco_beir"


def test_msmarco_training_metric_is_the_recall_ndcg_variant():
    # Its output equals compute_reward_metrics_ndcg's, so only identity tells the two routes apart.
    metric_fn = reward_registry.select_training_metrics("data/local_index_search/msmarco_health/sparse/train.parquet")

    assert metric_fn is reward_registry.compute_reward_metrics_recall_ndcg


def test_sql_all_train_files_select_the_sql_training_metric():
    # sql_all mixes bird and spider; its path names neither, so 'sql' is matched on its own.
    metric_fn = reward_registry.select_training_metrics("data/sql/sql_all/train.parquet")

    assert metric_fn is reward_registry.compute_reward_metrics_sql
