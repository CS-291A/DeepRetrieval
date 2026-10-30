"""A5: modules whose dependencies are not shipped must import, and fail clearly only when used.

The wikisql reward this file used to cover was removed with the rest of wikisql (upstream 8c8cbd7).
"""
import importlib

import pytest


def test_main_eval_import_succeeds_without_math_reward():
    module = importlib.import_module("verl.trainer.main_eval")

    assert hasattr(module, "math") is False


def test_main_eval_math_reward_raises_clear_import_error():
    module = importlib.import_module("verl.trainer.main_eval")

    with pytest.raises(ImportError, match=r"verl\.utils\.reward_score\.math"):
        module.select_reward_fn("lighteval/MATH")
