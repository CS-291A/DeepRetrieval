"""A11, B18: retrieval-mode selection and the dense prebuilt-index fallback in reward_score/msmarco.py."""
import pytest

MODULE = "verl.utils.reward_score.msmarco"
GOOD_RESPONSE = '<|im_start|>assistant <think>t</think><answer>{"query": "q"}</answer>'


class FakeFaissSearcher:
    def __init__(self):
        self.prebuilt_calls = []

    def from_prebuilt_index(self, prebuilt_index_name, query_encoder):
        self.prebuilt_calls.append((prebuilt_index_name, query_encoder))
        return self


class MissingPrebuiltFaissSearcher:
    """pyserini returns None (after printing) when it cannot resolve a prebuilt index name."""

    @staticmethod
    def from_prebuilt_index(prebuilt_index_name, query_encoder):
        return None


def test_compute_score_rejects_data_source_without_retrieval_mode(load_reward_module):
    msmarco = load_reward_module(MODULE)

    with pytest.raises(ValueError, match=r"'msmarco_test'.*'sparse' or 'dense'"):
        msmarco.compute_score(GOOD_RESPONSE, {"target": ["1"]}, data_source="msmarco_test")


def test_dense_fallback_passes_the_tct_colbert_query_encoder(load_reward_module, monkeypatch):
    msmarco = load_reward_module(MODULE)
    faiss = FakeFaissSearcher()
    monkeypatch.setattr(msmarco, "FaissSearcher", faiss)

    searcher = msmarco.get_searcher(mode="dense")

    assert searcher is faiss
    assert faiss.prebuilt_calls == [("msmarco-v1-passage.tct_colbert", "castorini/tct_colbert-msmarco")]


def test_dense_fallback_raises_when_prebuilt_index_is_unavailable(load_reward_module, monkeypatch):
    msmarco = load_reward_module(MODULE)
    monkeypatch.setattr(msmarco, "FaissSearcher", MissingPrebuiltFaissSearcher)

    with pytest.raises(RuntimeError, match=r"msmarco-v1-passage\.tct_colbert"):
        msmarco.get_searcher(mode="dense")
