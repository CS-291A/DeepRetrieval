"""BM25 eval scripts: the running recall summary reports an empty run instead of hiding it (audit item 2)."""
import pytest

from tests.deepretrieval.evals._loader import load_function

SCRIPTS = [
    "src/eval/BM25/nq.py",
    "src/eval/BM25/squad.py",
    "src/eval/BM25/triviaqa.py",
    "src/eval/BM25/nq_base.py",
    "src/eval/BM25/squad_base.py",
    "src/eval/BM25/triviaqa_base.py",
]


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_empty_recall_lists_report_no_scored_queries(relpath, capsys):
    summary = load_function(relpath, "print_recall_summary", {})

    summary([], [], [], [])

    assert capsys.readouterr().out == "Recall: no queries scored\n"


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_recall_summary_prints_the_same_numbers(relpath, capsys):
    summary = load_function(relpath, "print_recall_summary", {})

    summary([1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1])

    assert capsys.readouterr().out.splitlines() == [
        "Recall@1: 0.25",
        "Recall@5: 0.5",
        "Recall@20: 0.75",
        "Recall@100: 1.0",
    ]


def test_recall_lists_out_of_step_is_a_programming_error():
    summary = load_function(SCRIPTS[0], "print_recall_summary", {})

    with pytest.raises(AssertionError, match="out of step"):
        summary([1], [1, 0], [1], [1])
