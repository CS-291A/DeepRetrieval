"""`extract_solution` copies in the eval scripts that strip a chat-template header first."""
import re

import pytest

from tests.deepretrieval.evals._loader import load_function

HEADER_SCRIPTS = (
    "src/eval/BM25/msmarco.py",
    "src/eval/BM25/nq.py",
    "src/eval/BM25/squad.py",
    "src/eval/BM25/triviaqa.py",
    "src/eval/Dense/msmarco.py",
    "src/eval/SearchEngine/eval_ctgov.py",
    "src/eval/SearchEngine/eval_pubmed.py",
    "src/eval/SearchEngine/eval_pubmed_no_reason.py",
)


@pytest.fixture(params=HEADER_SCRIPTS)
def extract_solution(request):
    return load_function(request.param, "extract_solution", {"re": re})


def test_missing_header_returns_no_answer_and_the_raw_text(extract_solution):
    text = '<answer>{"query": "q"}</answer>'

    assert extract_solution(text) == (None, text)


def test_header_then_answer_returns_last_answer(extract_solution):
    text = 'user prompt <|im_start|>assistant <answer>a</answer> <answer>{"query": "q"}</answer>'

    answer, processed = extract_solution(text)

    assert answer == '{"query": "q"}'
    assert processed.startswith("<answer>a</answer>")
