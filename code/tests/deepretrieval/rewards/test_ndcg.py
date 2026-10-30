"""G4 (approved change): ndcg_at_k must stay in [0, 1] and reach 1.0 only for an ideal ranking."""
import pytest

from src.Lucene.utils import ndcg_at_k


def test_ndcg_ideal_dcg_uses_scores_sorted_descending():
    # qrels list 'a' (score 1) before 'b' (score 2); ranking b first is the ideal ranking.
    assert ndcg_at_k(["b", "a"], ["a", "b"], 10, rel_scores=[1, 2]) == pytest.approx(1.0)


def test_ndcg_counts_a_duplicate_retrieved_id_only_once():
    assert ndcg_at_k(["a", "a", "a"], "a", 10) == pytest.approx(1.0)


def test_ndcg_duplicate_keeps_its_rank_slot_with_zero_gain():
    # 'b' stays at rank 3 (the duplicate 'a' occupies rank 2), so this is worse than ['a', 'b'].
    expected = (1 + 1 / 2) / (1 + 1 / 1.584962500721156)

    assert ndcg_at_k(["a", "a", "b"], ["a", "b"], 10) == pytest.approx(expected)


@pytest.mark.parametrize(("retrieved", "target", "rel_scores"), [
    (["a", "b", "c"], ["a", "b", "c"], [3, 2, 1]),
    (["c", "a", "b"], ["b", "a", "c"], [1, 2, 3]),
    (["a", "x", "y"], ["a"], None),
    (["b", "a", "z"], ["a", "b"], None),
])
def test_ndcg_perfect_ranking_scores_one(retrieved, target, rel_scores):
    assert ndcg_at_k(retrieved, target, 10, rel_scores=rel_scores) == pytest.approx(1.0)


@pytest.mark.parametrize(("retrieved", "target", "rel_scores"), [
    (["b", "a"], ["a", "b"], [1, 2]),
    (["a", "b"], ["a", "b"], [1, 2]),
    (["a", "a", "a"], "a", None),
    (["x", "b", "b", "a", "a"], ["a", "b", "c"], [2, 2, 1]),
    (["x", "y"], ["a"], None),
    (["a", "b", "c", "d"], ["d", "c"], [5, 1]),
])
def test_ndcg_is_between_zero_and_one(retrieved, target, rel_scores):
    score = ndcg_at_k(retrieved, target, 3, rel_scores=rel_scores)

    assert 0.0 <= score <= 1.0 + 1e-12


def test_ndcg_list_target_without_scores_uses_binary_relevance():
    # One of two relevant docs at rank 2 (binary relevance for a list target without scores).
    expected = (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)

    assert ndcg_at_k(["x", "a", "y"], ["a", "b"], 10) == pytest.approx(expected)


def test_ndcg_nothing_relevant_retrieved_scores_zero():
    assert ndcg_at_k(["x", "y"], ["a", "b"], 10, rel_scores=[1, 1]) == pytest.approx(0.0)
