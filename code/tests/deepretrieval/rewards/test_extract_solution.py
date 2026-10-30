"""A1: extract_solution must not crash when the response has no assistant header."""
import pytest

REWARD_MODULES = [
    *(f"verl.utils.reward_score.{name}" for name in (
        "bird", "bird_no_reason", "ctgov", "fever", "fiqa", "hotpotqa", "msmarco", "msmarco_beir",
        "nfcorpus", "nq_no_reason", "nq_serini", "pmd_no_reason", "pubmed", "scifact", "screening",
        "spider", "spider_no_reason", "squad", "squad_no_reason", "tqa_no_reason", "triviaqa", "wikisql")),
    *(f"verl.utils.reward_score_dense.{name}" for name in (
        "fever", "hotpotqa", "msmarco_beir", "nfcorpus", "scifact")),
]


@pytest.mark.parametrize("module_name", REWARD_MODULES)
def test_extract_solution_without_header_returns_no_answer_and_empty_text(load_reward_module, module_name):
    module = load_reward_module(module_name)

    result = module.extract_solution("<think>x</think><answer>{}</answer>")

    assert result == (None, "")
