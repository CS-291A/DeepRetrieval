"""Audit 3, item 1: screening.validate_criteria_design catches only malformed-JSON errors.

The bare `except:` around `extract_json_from_llm_output` also swallowed KeyboardInterrupt and
SystemExit. Malformed criteria JSON must still be rejected with (False, None).
"""
import pytest

from verl.utils.reward_score import screening

DEEP = 100_000


@pytest.mark.parametrize("criteria", [
    "no braces at all",                     # ValueError('No JSON structure found.')
    "{not: valid}",                         # json.JSONDecodeError via the backup regex
    "```json\n{\"a\": }\n```",              # json.JSONDecodeError via the fenced block
    "{\"a\": " + "[" * DEEP + "]" * DEEP + "}",  # RecursionError from json.loads on deep nesting
])
def test_malformed_criteria_json_is_rejected(criteria):
    assert screening.validate_criteria_design(f"<criteria>{criteria}</criteria>") == (False, None)


def test_valid_criteria_json_is_accepted():
    criteria = '{"c1": "adult patients only", "c2": "randomized controlled trial"}'

    assert screening.validate_criteria_design(f"<criteria>{criteria}</criteria>") == (True, 2)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupt_in_criteria_parser_propagates(monkeypatch, interrupt):
    def parser(text):
        raise interrupt()

    monkeypatch.setattr(screening, "extract_json_from_llm_output", parser)

    with pytest.raises(interrupt):
        screening.validate_criteria_design("<criteria>{\"c1\": \"x y\"}</criteria>")
