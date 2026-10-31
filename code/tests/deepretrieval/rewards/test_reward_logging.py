"""Audit 3, item 6: reward functions and search clients log through module loggers, not print.

Per-rollout chatter (each query, recall, structure validation, malformed-answer notes) is DEBUG; the
`do_print`-sampled sample dumps are INFO and keep their sampling; infrastructure problems are WARNING.
Nothing in these packages writes to stdout outside an `if __name__ == "__main__":` demo block.
"""
import ast
import logging
import pathlib
import sys
import types

import numpy as np
import pytest

import verl

from .conftest import FakeDenseSearch, FakeMultiFieldSearch

PACKAGE_ROOT = pathlib.Path(verl.__file__).parent
LOGGED_PACKAGES = ["utils/reward_score", "utils/reward_score_dense", "utils/apis"]
MALFORMED = "<|im_start|>assistant <think>t</think> no answer tags here"


class FakeSearchAPI:
    def __init__(self, ids):
        self.ids = ids

    def search_with_keywords(self, query, topk=-1):
        return self.ids


def _pubmed(module):
    return module.compute_score(MALFORMED, {"target": ["1"]}, search_api=FakeSearchAPI(["1"]), pub_date="2020/01/01")


def _nq(module):
    return module.compute_score(MALFORMED, {"target": np.array(["Paris"])})


def _sparse_scifact(module):
    module.search_system = FakeMultiFieldSearch(["d1"])
    return module.compute_score(MALFORMED, {"target": "d1"}, data_source="scifact")


def _dense_fever(module):
    module.search_system = FakeDenseSearch(["d1"])
    return module.compute_score(MALFORMED, {"target": ["d1"], "score": [1]}, data_source="fever")


def _bird(module):
    return module.compute_score(MALFORMED, {"target": "SELECT 1"}, data_source="bird", db_path="unused.sqlite")


REPRESENTATIVE = [
    ("verl.utils.reward_score.pubmed", _pubmed),
    ("verl.utils.reward_score.nq_serini", _nq),
    ("verl.utils.reward_score.scifact", _sparse_scifact),
    ("verl.utils.reward_score_dense.fever", _dense_fever),
    ("verl.utils.reward_score.bird", _bird),
]


def _force_sampling(monkeypatch, module, sampled):
    """Make `random.randint(1, N) == 1` (the do_print draw) come out as requested."""
    monkeypatch.setattr(module, "random", types.SimpleNamespace(randint=lambda low, high: 1 if sampled else 2))


def _records(caplog, module_name, level):
    return [r.getMessage() for r in caplog.records if r.name == module_name and r.levelno == level]


@pytest.mark.parametrize(("module_name", "score_malformed"), REPRESENTATIVE)
def test_malformed_answer_is_logged_at_debug(load_reward_module, monkeypatch, caplog, module_name,
                                             score_malformed):
    module = load_reward_module(module_name)
    _force_sampling(monkeypatch, module, sampled=False)

    with caplog.at_level(logging.DEBUG, logger=module_name):
        score_malformed(module)

    assert any("No valid answer tags found" in m for m in _records(caplog, module_name, logging.DEBUG))


@pytest.mark.parametrize(("module_name", "score_malformed"), REPRESENTATIVE)
def test_sampled_rollout_writes_nothing_to_stdout(load_reward_module, monkeypatch, capsys, caplog, module_name,
                                                  score_malformed):
    module = load_reward_module(module_name)
    _force_sampling(monkeypatch, module, sampled=True)

    with caplog.at_level(logging.DEBUG, logger=module_name):
        score_malformed(module)

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(("module_name", "score_malformed"), REPRESENTATIVE)
def test_sampled_rollout_dumps_final_score_at_info(load_reward_module, monkeypatch, caplog, module_name,
                                                   score_malformed):
    module = load_reward_module(module_name)
    _force_sampling(monkeypatch, module, sampled=True)

    with caplog.at_level(logging.DEBUG, logger=module_name):
        total = score_malformed(module)

    assert any(f"Total: {total}" in m for m in _records(caplog, module_name, logging.INFO))


@pytest.mark.parametrize(("module_name", "score_malformed"), REPRESENTATIVE)
def test_unsampled_rollout_logs_nothing_at_info(load_reward_module, monkeypatch, caplog, module_name,
                                                score_malformed):
    module = load_reward_module(module_name)
    _force_sampling(monkeypatch, module, sampled=False)
    caplog.clear()  # drop import-time records such as "index not found"

    with caplog.at_level(logging.DEBUG, logger=module_name):
        score_malformed(module)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO] == []


@pytest.mark.parametrize("module_name", ["verl.utils.reward_score.pubmed", "verl.utils.reward_score.pmd_no_reason"])
def test_pubmed_query_is_logged_at_debug(load_reward_module, caplog, module_name):
    module = load_reward_module(module_name)

    with caplog.at_level(logging.DEBUG, logger=module_name):
        module.calculate_answer_score('{"query": "asthma AND child"}', ["1"], search_api=FakeSearchAPI(["1"]),
                                      pub_date="2020/01/01")

    assert [m for m in _records(caplog, module_name, logging.DEBUG) if "Query:" in m] == [
        "Query: asthma AND child&datetype=pdat&mindate=1970/01/01&maxdate=2020/01/01"]


@pytest.mark.parametrize("module_name", ["verl.utils.reward_score.pubmed", "verl.utils.reward_score.pmd_no_reason"])
def test_pubmed_query_is_not_written_to_stdout(load_reward_module, capsys, module_name):
    module = load_reward_module(module_name)

    module.calculate_answer_score('{"query": "asthma"}', ["1"], search_api=FakeSearchAPI(["1"]), pub_date="2020/01/01")

    assert capsys.readouterr().out == ""


def test_missing_java_init_is_logged_at_warning(load_reward_module, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "utils.java_init", None)  # makes the import raise ImportError
    module_name = "verl.utils.reward_score.nq_serini"

    with caplog.at_level(logging.DEBUG, logger=module_name):
        load_reward_module(module_name)

    assert _records(caplog, module_name, logging.WARNING) == ["Failed to import java_init"]


def test_missing_index_is_logged_at_warning(load_reward_module, caplog):
    module_name = "verl.utils.reward_score.scifact"

    with caplog.at_level(logging.DEBUG, logger=module_name):
        load_reward_module(module_name)  # imported from an empty cwd, so no index exists

    messages = _records(caplog, module_name, logging.WARNING)
    assert len(messages) == 1
    assert messages[0].startswith("Pyserini index not found for scifact at ")  # + the resolved path and data-dir var


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _prints_outside_main(path):
    tree = ast.parse(path.read_text())
    in_main = {id(n) for top in tree.body if _is_main_guard(top) for n in ast.walk(top)}
    return [f"{path.relative_to(PACKAGE_ROOT)}:{n.lineno}" for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
            and id(n) not in in_main]


LOGGED_FILES = sorted(p for pkg in LOGGED_PACKAGES for p in (PACKAGE_ROOT / pkg).rglob("*.py"))


def test_logged_packages_are_found():
    assert len(LOGGED_FILES) >= 28


@pytest.mark.parametrize("path", LOGGED_FILES, ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_no_print_outside_main_block(path):
    assert _prints_outside_main(path) == []
