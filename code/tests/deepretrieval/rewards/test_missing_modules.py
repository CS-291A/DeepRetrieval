"""A5: modules whose dependencies are not shipped must import, and fail clearly only when used."""
import importlib

import pytest

WIKISQL_RESPONSE = '<|im_start|>assistant <think>t</think><answer>{"sql": "SELECT 1"}</answer>'


def test_wikisql_reward_import_succeeds_without_src_sql_wikisql(load_reward_module):
    module = load_reward_module("verl.utils.reward_score.wikisql")

    assert module.check_json_format('{"sql": "SELECT 1"}') is True


def test_wikisql_reward_raises_clear_import_error_when_scoring(load_reward_module):
    module = load_reward_module("verl.utils.reward_score.wikisql")

    with pytest.raises(ImportError, match=r"src\.sql\.wikisql"):
        module.compute_score(WIKISQL_RESPONSE, {"target": "SELECT 1"}, "wikisql", "db.sqlite", "db")


def test_main_eval_import_succeeds_without_math_reward():
    module = importlib.import_module("verl.trainer.main_eval")

    assert hasattr(module, "math") is False


def test_main_eval_math_reward_raises_clear_import_error():
    module = importlib.import_module("verl.trainer.main_eval")

    with pytest.raises(ImportError, match=r"verl\.utils\.reward_score\.math"):
        module.select_reward_fn("lighteval/MATH")
