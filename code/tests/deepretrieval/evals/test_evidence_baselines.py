"""src/eval/baselines/evidence_gpt_claude{,_no_reason}.py."""
import ast
import re
from typing import cast

import pytest

from tests.deepretrieval.evals._loader import CODE_DIR, load_function, load_script

EVIDENCE_SCRIPTS = (
    "src/eval/baselines/evidence_gpt_claude.py",
    "src/eval/baselines/evidence_gpt_claude_no_reason.py",
)


def _claude_llm_names(relpath):
    """Every constant passed as `llm=` to get_claude_response in the script."""
    tree = ast.parse((CODE_DIR / relpath).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "get_claude_response":
            names += [cast(ast.Constant, kw.value).value for kw in node.keywords if kw.arg == "llm"]
    return names


@pytest.fixture
def bedrock_models(monkeypatch):
    claude_api = load_script(monkeypatch, "src/utils/claude_api.py", ("boto3",))
    return claude_api.BEDROCK_MODEL_NAME_MAP


@pytest.mark.parametrize("relpath", EVIDENCE_SCRIPTS)
def test_every_claude_model_name_is_accepted_by_get_claude_response(bedrock_models, relpath):
    names = _claude_llm_names(relpath)

    assert sorted(names) == ["haiku", "sonnet"]
    assert set(names) <= set(bedrock_models)


class _InterruptingJson:
    """json stand-in whose loads() is interrupted, as by Ctrl-C mid-parse."""

    @staticmethod
    def loads(text):
        raise KeyboardInterrupt


@pytest.mark.parametrize("relpath", EVIDENCE_SCRIPTS)
def test_query_extraction_does_not_swallow_keyboard_interrupt(relpath):
    extract = load_function(relpath, "extract_json_from_llm_output",
                            {"re": re, "json": _InterruptingJson})

    with pytest.raises(KeyboardInterrupt, match="^$"):
        extract('reasoning {"query": "q"}')
