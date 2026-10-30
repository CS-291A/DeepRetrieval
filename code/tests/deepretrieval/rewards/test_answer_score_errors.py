"""B17 / A6 / D29: a model-written query never crashes training.

Misconfiguration (no search client / index, empty label) fails fast. Malformed model output and any
failure while executing the model's query earn the same constant HEAD returned; execution failures are
logged at ERROR with a traceback.

Covers the search-engine rewards (PubMed / CT.gov), the NDCG rewards (sparse and dense BEIR tasks,
msmarco) and the answer-span rewards (nq / squad / triviaqa).
"""
import json
import logging
import sys
from unittest import mock

import numpy as np
import pytest

from verl.utils.apis import ctgov as ctgov_api
from verl.utils.apis import pubmed as pubmed_api

from .conftest import FakeDenseSearch, FakeMultiFieldSearch

BAD_QUERY_OUTPUTS = ["not json", '{"q": "x"}', '["query"]', '"query"']


class FailingBackend:
    """A search client whose every call fails like an unreachable index or API."""

    def _fail(self, *args, **kwargs):
        raise RuntimeError("backend down")

    search = batch_search = search_with_keywords = _fail


def _logged_failures(caplog, module_name):
    """ERROR records from the reward module's logger that carry a traceback."""
    return [r.getMessage() for r in caplog.records
            if r.name == module_name and r.levelno == logging.ERROR and r.exc_info]


# ---------------------------------------------------------------- PubMed / CT.gov


class FakeSearchAPI:
    def __init__(self, ids):
        self.ids = ids
        self.queries = []

    def search_with_keywords(self, query, topk=-1):
        self.queries.append(query)
        return self.ids


API_REWARDS = [
    ("verl.utils.reward_score.pubmed", "publication"),
    ("verl.utils.reward_score.pmd_no_reason", "publication"),
    ("verl.utils.reward_score.ctgov", "trial"),
]


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
@pytest.mark.parametrize("answer_text", BAD_QUERY_OUTPUTS)
def test_api_reward_scores_malformed_query_as_minus_four(load_reward_module, module_name, literature_type,
                                                         answer_text):
    module = load_reward_module(module_name)
    api = FakeSearchAPI(["1"])

    score = module.calculate_answer_score(answer_text, ["1"], search_api=api, literature_type=literature_type,
                                          pub_date="2020/01/01")

    assert (score, api.queries) == (-4, [])


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
def test_api_reward_scores_full_recall_as_five(load_reward_module, module_name, literature_type):
    module = load_reward_module(module_name)

    score = module.calculate_answer_score('{"query": "q"}', np.array(["1", "2"]), search_api=FakeSearchAPI(["2", "1"]),
                                          literature_type=literature_type, pub_date="2020/01/01")

    assert score == 5


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
def test_api_reward_scores_search_api_failure_as_minus_four(load_reward_module, module_name, literature_type):
    module = load_reward_module(module_name)

    score = module.calculate_answer_score('{"query": "q"}', ["1"], search_api=FailingBackend(),
                                          literature_type=literature_type, pub_date="2020/01/01")

    assert score == -4


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
def test_api_reward_logs_search_api_failure_with_traceback(load_reward_module, caplog, module_name,
                                                           literature_type):
    module = load_reward_module(module_name)

    with caplog.at_level(logging.ERROR, logger=module_name):
        module.calculate_answer_score('{"query": "weird query"}', ["1"], search_api=FailingBackend(),
                                      literature_type=literature_type, pub_date="2020/01/01")

    assert len(_logged_failures(caplog, module_name)) == 1
    assert "weird query" in _logged_failures(caplog, module_name)[0]


def test_pubmed_reward_scores_pubmed_error_payload_as_minus_four(load_reward_module, caplog):
    module = load_reward_module("verl.utils.reward_score.pubmed")
    api = pubmed_api.PubmedAPI(retry=1, request_delay=0)
    error_payload = mock.Mock(status_code=200, text='{"esearchresult": {"ERROR": "Invalid query"}}')

    with mock.patch.object(pubmed_api.requests, "get", autospec=True, return_value=error_payload), \
            caplog.at_level(logging.ERROR, logger=module.__name__):
        score = module.calculate_answer_score('{"query": "q"}', ["1"], search_api=api, pub_date="2020/01/01")

    assert (score, len(_logged_failures(caplog, module.__name__))) == (-4, 1)


def test_ctgov_reward_scores_api_failing_every_retry_as_minus_four(load_reward_module, caplog):
    module = load_reward_module("verl.utils.reward_score.ctgov")
    api = ctgov_api.CTGovAPI(retry=1, request_delay=0)
    down = mock.Mock(status_code=503, text="unavailable")

    with mock.patch.object(ctgov_api.requests, "get", autospec=True, return_value=down) as get, \
            caplog.at_level(logging.ERROR, logger=module.__name__):
        score = module.calculate_answer_score('{"query": "asthma"}', ["NCT1"], search_api=api)

    assert (score, get.call_count, len(_logged_failures(caplog, module.__name__))) == (-4, 2, 1)


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
def test_api_reward_requires_a_search_api(load_reward_module, module_name, literature_type):
    module = load_reward_module(module_name)

    with pytest.raises(AssertionError, match="search_api"):
        module.calculate_answer_score('{"query": "q"}', ["1"], search_api=None, literature_type=literature_type,
                                      pub_date="2020/01/01")


@pytest.mark.parametrize(("module_name", "literature_type"), API_REWARDS)
def test_api_reward_rejects_empty_label(load_reward_module, module_name, literature_type):
    module = load_reward_module(module_name)

    with pytest.raises(ValueError, match="empty label"):
        module.calculate_answer_score('{"query": "q"}', np.array([]), search_api=FakeSearchAPI(["1"]),
                                      literature_type=literature_type, pub_date="2020/01/01")


# ---------------------------------------------------------------- NDCG rewards (BEIR tasks)

SPARSE_NDCG = ["fever", "fiqa", "hotpotqa", "msmarco_beir", "nfcorpus", "scifact"]
DENSE_NDCG = ["fever", "hotpotqa", "msmarco_beir", "nfcorpus", "scifact"]
NDCG_REWARDS = ([(f"verl.utils.reward_score.{name}", FakeMultiFieldSearch) for name in SPARSE_NDCG]
                + [(f"verl.utils.reward_score_dense.{name}", FakeDenseSearch) for name in DENSE_NDCG])


def _ndcg_score(module, json_str, docid):
    """Score against one relevant doc, in the label shape compute_score builds for each task."""
    # scifact: compute_score passes str(target) (one docid per row) and no relevance scores.
    if module.__name__.endswith("scifact"):
        return module.calculate_answer_score(json_str, docid, 10)
    label = [docid] if docid else []
    return module.calculate_answer_score(json_str, label, [1] * len(label), 10)


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
@pytest.mark.parametrize("answer_text", BAD_QUERY_OUTPUTS)
def test_ndcg_reward_scores_malformed_query_as_minus_two(load_reward_module, monkeypatch, module_name, fake_search,
                                                         answer_text):
    module = load_reward_module(module_name)
    backend = fake_search(["d1"])
    monkeypatch.setattr(module, "search_system", backend)

    score = _ndcg_score(module, answer_text, "d1")

    assert (score, backend.queries) == (-2, [])


def test_sparse_ndcg_reward_still_executes_a_non_string_query_like_head(load_reward_module, monkeypatch):
    # HEAD sent {"query": 5} to the searcher (as "contents:5") and scored the hits; keep that value.
    module = load_reward_module("verl.utils.reward_score.fever")
    backend = FakeMultiFieldSearch(["d1"])
    monkeypatch.setattr(module, "search_system", backend)

    score = module.calculate_answer_score('{"query": 5}', ["d1"], [1], 10)

    assert (score, backend.queries) == (pytest.approx(1.0), [5])


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
def test_ndcg_reward_scores_perfect_retrieval_as_one(load_reward_module, monkeypatch, module_name, fake_search):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", fake_search(["d1", "x"]))

    score = _ndcg_score(module, '{"query": "q"}', "d1")

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
def test_ndcg_reward_scores_search_failure_as_minus_two(load_reward_module, monkeypatch, module_name, fake_search):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", FailingBackend())

    assert _ndcg_score(module, '{"query": "q"}', "d1") == -2


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
def test_ndcg_reward_logs_search_failure_with_traceback(load_reward_module, monkeypatch, caplog, module_name,
                                                        fake_search):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", FailingBackend())

    with caplog.at_level(logging.ERROR, logger=module_name):
        _ndcg_score(module, '{"query": "weird query"}', "d1")

    assert len(_logged_failures(caplog, module_name)) == 1
    assert "weird query" in _logged_failures(caplog, module_name)[0]


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
def test_ndcg_reward_raises_when_index_was_not_loaded(load_reward_module, module_name, fake_search):
    module = load_reward_module(module_name)  # imported from an empty cwd, so no index was found

    with pytest.raises(RuntimeError, match="index"):
        _ndcg_score(module, '{"query": "q"}', "d1")


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
def test_ndcg_reward_rejects_empty_label(load_reward_module, monkeypatch, module_name, fake_search):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", fake_search(["d1"]))

    with pytest.raises(ValueError, match="empty label"):
        _ndcg_score(module, '{"query": "q"}', "")


@pytest.mark.parametrize(("module_name", "fake_search"), NDCG_REWARDS)
@pytest.mark.parametrize(("answer_text", "expected"), [
    ("5", -2), ("null", -2),  # HEAD crashed with TypeError in check_json_format; now a format failure
    ('["query"]', 0),  # HEAD value: format passes, answer -2, total clipped to 0
])
def test_ndcg_reward_scores_non_object_json_like_head_without_crashing(load_reward_module, monkeypatch, module_name,
                                                                       fake_search, answer_text, expected):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", fake_search(["d1"]))
    response = f"<|im_start|>assistant <think>t</think><answer>{answer_text}</answer>"

    target = "d1" if module_name.endswith("scifact") else ["d1"]

    score = module.compute_score(response, {"target": target, "score": [1]}, data_source="test")

    assert score == expected


@pytest.mark.parametrize(("module_name", "fake_search"), [
    ("verl.utils.reward_score.scifact", FakeMultiFieldSearch),
    ("verl.utils.reward_score_dense.scifact", FakeDenseSearch),
])
def test_scifact_reward_counts_a_duplicated_hit_once(load_reward_module, monkeypatch, module_name, fake_search):
    # G4: HEAD scored this 0.1 + 1.63 (the duplicate 'd1' gained twice); NDCG now caps at 1.
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "search_system", fake_search(["d1", "d1", "x"]))
    response = '<|im_start|>assistant <think>t</think><answer>{"query": "q"}</answer>'

    score = module.compute_score(response, {"target": "d1"}, data_source="scifact_test")

    assert score == pytest.approx(1.1)


# ---------------------------------------------------------------- msmarco (Lucene / Faiss hits)


class FakeHit:
    def __init__(self, raw):
        self.lucene_document = self
        self.raw = raw
        self.docid = raw

    def get(self, field):
        assert field == "raw"
        return self.raw


class FakeLuceneSearcher:
    def __init__(self, raws):
        self.raws = raws
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        return [FakeHit(raw) for raw in self.raws[:k]]


def _msmarco(load_reward_module, monkeypatch, searcher):
    module = load_reward_module("verl.utils.reward_score.msmarco")
    monkeypatch.setattr(module, "_searcher", searcher)
    return module


@pytest.mark.parametrize("answer_text", BAD_QUERY_OUTPUTS)
def test_msmarco_reward_scores_malformed_query_as_minus_two(load_reward_module, monkeypatch, answer_text):
    searcher = FakeLuceneSearcher([json.dumps({"id": "1"})])
    module = _msmarco(load_reward_module, monkeypatch, searcher)

    score = module.calculate_answer_score(answer_text, ["1"], [1], 10, 10, mode="sparse")

    assert (score, searcher.queries) == (-2, [])


def test_msmarco_reward_scores_perfect_sparse_retrieval_as_one(load_reward_module, monkeypatch):
    module = _msmarco(load_reward_module, monkeypatch, FakeLuceneSearcher([json.dumps({"id": "1"})]))

    score = module.calculate_answer_score('{"query": "q"}', [1], [1], 10, 10, mode="sparse")

    assert score == pytest.approx(1.0)


def test_msmarco_reward_scores_search_failure_as_minus_two(load_reward_module, monkeypatch, caplog):
    module = _msmarco(load_reward_module, monkeypatch, FailingBackend())

    with caplog.at_level(logging.ERROR, logger=module.__name__):
        score = module.calculate_answer_score('{"query": "q"}', ["1"], [1], 10, 10, mode="sparse")

    assert (score, len(_logged_failures(caplog, module.__name__))) == (-2, 1)


def test_msmarco_reward_rejects_empty_label(load_reward_module, monkeypatch):
    module = _msmarco(load_reward_module, monkeypatch, FakeLuceneSearcher([]))

    with pytest.raises(ValueError, match="empty label"):
        module.calculate_answer_score('{"query": "q"}', [], [], 10, 10, mode="sparse")


# ---------------------------------------------------------------- answer-span rewards (wikipedia-dpr)

SPAN_REWARDS = [
    ("verl.utils.reward_score.nq_serini", -4),
    ("verl.utils.reward_score.nq_no_reason", -4),
    ("verl.utils.reward_score.triviaqa", -4),
    ("verl.utils.reward_score.tqa_no_reason", -4),
    ("verl.utils.reward_score.squad", -3),
    ("verl.utils.reward_score.squad_no_reason", -3),
]


def _contents(text):
    return json.dumps({"contents": text})


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
@pytest.mark.parametrize("answer_text", BAD_QUERY_OUTPUTS)
def test_span_reward_scores_malformed_query_as_error_constant(load_reward_module, monkeypatch, module_name,
                                                              error_score, answer_text):
    module = load_reward_module(module_name)
    searcher = FakeLuceneSearcher([_contents("Paris is the capital")])
    monkeypatch.setattr(module, "_searcher", searcher)

    score = module.calculate_answer_score_scale(answer_text, ["Paris"])

    assert (score, searcher.queries) == (error_score, [])


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_parses_hits_lazily_and_stops_at_first_answer(load_reward_module, monkeypatch, module_name,
                                                                  error_score):
    module = load_reward_module(module_name)
    # Only the first hit is valid JSON: parsing past the answering hit would raise.
    monkeypatch.setattr(module, "_searcher", FakeLuceneSearcher([_contents("Paris is the capital"), "{broken"]))

    score = module.calculate_answer_score_scale('{"query": "capital of France"}', ["Paris"])

    assert score == 5


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_ranks_answer_by_first_matching_hit(load_reward_module, monkeypatch, module_name, error_score):
    module = load_reward_module(module_name)
    hits = [_contents(f"filler {i}") for i in range(30)] + [_contents("Paris")]
    monkeypatch.setattr(module, "_searcher", FakeLuceneSearcher(hits))

    score = module.calculate_answer_score_scale('{"query": "q"}', ["Paris"])

    assert score == 2  # rank 31 falls in the (20, 50] band


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_scores_search_failure_as_error_constant(load_reward_module, monkeypatch, module_name,
                                                            error_score):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "_searcher", FailingBackend())

    assert module.calculate_answer_score_scale('{"query": "q"}', ["Paris"]) == error_score


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_logs_search_failure_with_traceback(load_reward_module, monkeypatch, caplog, module_name,
                                                        error_score):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "_searcher", FailingBackend())

    with caplog.at_level(logging.ERROR, logger=module_name):
        module.calculate_answer_score_scale('{"query": "weird query"}', ["Paris"])

    assert len(_logged_failures(caplog, module_name)) == 1
    assert "weird query" in _logged_failures(caplog, module_name)[0]


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_rejects_empty_label(load_reward_module, monkeypatch, module_name, error_score):
    module = load_reward_module(module_name)
    monkeypatch.setattr(module, "_searcher", FakeLuceneSearcher([_contents("Paris")]))

    with pytest.raises(ValueError, match="empty label"):
        module.calculate_answer_score_scale('{"query": "q"}', [])


# ---------------------------------------------------------------- utils.java_init import guard


class _RaisingFinder:
    """Meta-path finder that makes `import utils.java_init` raise the given exception."""

    def __init__(self, exc):
        self.exc = exc

    def find_spec(self, name, path=None, target=None):
        if name == "utils.java_init":
            raise self.exc


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_import_tolerates_missing_java_init(load_reward_module, monkeypatch, module_name, error_score):
    monkeypatch.setitem(sys.modules, "utils.java_init", None)  # makes the import raise ImportError

    module = load_reward_module(module_name)

    assert module.calculate_answer_score_scale("not json", ["Paris"]) == error_score


@pytest.mark.parametrize(("module_name", "error_score"), SPAN_REWARDS)
def test_span_reward_import_propagates_java_init_failure(load_reward_module, monkeypatch, module_name, error_score):
    monkeypatch.delitem(sys.modules, "utils.java_init", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_RaisingFinder(RuntimeError("JVM broken")), *sys.meta_path])

    with pytest.raises(RuntimeError, match="JVM broken"):
        load_reward_module(module_name)
