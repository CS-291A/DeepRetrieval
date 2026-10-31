"""GPT/Claude evidence baselines: the running recall summary reports an empty run instead of hiding it.

Same defect as audit item 2 in the BM25 scripts: `except Exception: continue` swallowed the ZeroDivisionError.
"""
import pytest

from tests.deepretrieval.evals._loader import load_function

SCRIPTS = [
    "src/eval/baselines/evidence_gpt_claude.py",
    "src/eval/baselines/evidence_gpt_claude_no_reason.py",
]


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_empty_recall_lists_report_no_scored_queries(relpath, capsys):
    summary = load_function(relpath, "print_recall_summary", {})

    summary([], [], [], [])

    assert capsys.readouterr().out == "Current Recall: no queries scored\n"


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_recall_summary_prints_the_same_numbers(relpath, capsys):
    summary = load_function(relpath, "print_recall_summary", {})

    summary([1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1])

    assert capsys.readouterr().out.splitlines() == [
        "Current Recall@1: 0.25",
        "Current Recall@5: 0.5",
        "Current Recall@20: 0.75",
        "Current Recall@100: 1.0",
    ]


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_recall_lists_out_of_step_is_a_programming_error(relpath):
    summary = load_function(relpath, "print_recall_summary", {})

    with pytest.raises(AssertionError, match="out of step"):
        summary([1], [1, 0], [1], [1])
