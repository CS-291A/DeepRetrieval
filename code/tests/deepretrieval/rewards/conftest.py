"""Fixtures for importing reward modules without a JVM, a retrieval index, or the network.

Every sparse/dense reward module imports pyserini (which starts a JVM at import time) or a
`src/{Lucene,Dense}/<task>/search.py` wrapper around it. `load_reward_module` replaces those
with inert stubs in `sys.modules`, runs the import from an empty cwd (so the module-level
`os.path.exists("data/...")` index checks see nothing), and removes everything afterwards.
Tests then inject their own fake search backend with `monkeypatch.setattr`.
"""
import importlib
import sys
import types

import pytest


class UnavailableBackend:
    """Stands in for every pyserini / faiss class. Reaching it means a test hit real retrieval."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("test reached a real retrieval backend")

    @classmethod
    def from_prebuilt_index(cls, *args, **kwargs):
        raise AssertionError("test reached a real prebuilt index download")


class FakeSimpleTokenizer:
    """Placeholder for pyserini's SimpleTokenizer; `fake_has_answers` ignores it."""


def fake_has_answers(text, answers, tokenizer, regex=False):
    """Case-insensitive substring match, standing in for pyserini's DPR answer matcher."""
    return any(answer.lower() in text.lower() for answer in answers)


_LUCENE_TASKS = ("fever", "fiqa", "hotpotqa", "msmarco_beir", "nfcorpus", "scifact")
_DENSE_TASKS = ("fever", "hotpotqa", "msmarco_beir", "nfcorpus")

_STUB_MODULES = {
    "pyserini": {},
    "pyserini.search": {},
    "pyserini.search.lucene": {"LuceneSearcher": UnavailableBackend},
    "pyserini.search.faiss": {"FaissSearcher": UnavailableBackend, "AutoQueryEncoder": UnavailableBackend},
    "pyserini.eval": {},
    "pyserini.eval.evaluate_dpr_retrieval": {"has_answers": fake_has_answers,
                                             "SimpleTokenizer": FakeSimpleTokenizer},
    "src.Dense.scifact.search": {"FaissHNSWSearcher": UnavailableBackend},
    **{f"src.Lucene.{task}.search": {"PyseriniMultiFieldSearch": UnavailableBackend} for task in _LUCENE_TASKS},
    **{f"src.Dense.{task}.search": {"PyseriniFaissSearcher": UnavailableBackend} for task in _DENSE_TASKS},
}


@pytest.fixture
def load_reward_module(monkeypatch, tmp_path, request):
    """Return a function that freshly imports a reward module against stubbed retrieval backends."""

    def load(module_name):
        for stub_name, attributes in _STUB_MODULES.items():
            stub = types.ModuleType(stub_name)
            for attribute, value in attributes.items():
                setattr(stub, attribute, value)
            monkeypatch.setitem(sys.modules, stub_name, stub)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        request.addfinalizer(lambda: sys.modules.pop(module_name, None))
        return importlib.import_module(module_name)

    return load


class FakeMultiFieldSearch:
    """Mimics src/Lucene/<task>/search.py: batch_search returns {query: [(docid, text, score), ...]}."""

    def __init__(self, docids):
        self.docids = list(docids)
        self.queries = []

    def batch_search(self, queries, top_k=10, threads=4):
        self.queries.extend(queries)
        return {query: [(docid, "text", 1.0) for docid in self.docids[:top_k]] for query in queries}


class FakeDenseSearch:
    """Mimics src/Dense/<task>/search.py: search returns [(docid, score), ...]."""

    def __init__(self, docids):
        self.docids = list(docids)
        self.queries = []

    def search(self, query, top_k=5, threads=8):
        self.queries.append(query)
        return [(docid, 1.0) for docid in self.docids[:top_k]]
