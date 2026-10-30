"""F40: unreferenced reward modules are removed; the API-key directory stays."""
import importlib.util
from pathlib import Path

import pytest

import verl.utils.reward_score as reward_score_pkg

REMOVED_MODULES = [
    "verl.utils.reward_score.msmarco_",
    "verl.utils.reward_score.utils",
    "verl.utils.reward_score.apis.pubmed",
    "verl.utils.reward_score.apis.ctgov",
]


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_dead_reward_module_is_removed(module_name):
    assert importlib.util.find_spec(module_name) is None


def test_api_key_directory_is_kept_for_pubmed_key():
    apis_dir = Path(reward_score_pkg.__file__).parent / "apis"

    assert (apis_dir / ".gitkeep").is_file()


def test_pubmed_reward_has_no_ctgov_stub(load_reward_module):
    pubmed = load_reward_module("verl.utils.reward_score.pubmed")

    assert not hasattr(pubmed, "run_search_ctgov")
