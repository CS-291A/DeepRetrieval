"""src/query_rewrite.py: client for the vLLM server started by vllm_host.sh."""
import re
import types

import pytest
import requests

from tests.deepretrieval.evals._loader import CODE_DIR, load_script

ANSWER = '<think>t</think><answer>{"query": "(a OR b)"}</answer>'


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def rewrite(monkeypatch):
    return load_script(monkeypatch, "src/query_rewrite.py")


@pytest.fixture
def posts(rewrite, monkeypatch):
    """Record each requests.post call and answer it with ANSWER."""
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(ANSWER)

    fake = types.SimpleNamespace(post=post, exceptions=requests.exceptions)
    monkeypatch.setattr(rewrite, "requests", fake)
    return calls


def _hosted_model():
    script = (CODE_DIR.parent / "vllm_host.sh").read_text()
    return re.search(r"^\s*--model (\S+)", script, re.MULTILINE).group(1)


def test_default_model_is_the_one_vllm_host_serves(rewrite, posts):
    rewrite.rewrite_query("q")

    assert posts[0][1]["json"]["model"] == _hosted_model()


def test_request_has_a_timeout(rewrite, posts):
    rewrite.rewrite_query("q")

    assert posts[0][1]["timeout"] == 120


def test_model_flag_is_sent_to_the_server(rewrite, posts, monkeypatch, capsys):
    monkeypatch.setattr(rewrite.sys, "argv", ["query_rewrite.py", "--query", "q", "--model", "m/x"])

    rewrite.main()

    assert posts[0][1]["json"]["model"] == "m/x"
    assert "Rewritten query: (a OR b)" in capsys.readouterr().out


def test_request_failure_keeps_the_original_cause(rewrite, monkeypatch):
    def post(url, **kwargs):
        raise requests.exceptions.ConnectionError("refused")  # audit: ignore[NETWORK] builds an exception object; no request is sent

    monkeypatch.setattr(rewrite, "requests",
                        types.SimpleNamespace(post=post, exceptions=requests.exceptions))

    with pytest.raises(RuntimeError, match="^API request failed: refused$") as info:
        rewrite.rewrite_query("q")

    assert isinstance(info.value.__cause__, requests.exceptions.ConnectionError)


def test_main_exits_nonzero_on_error(rewrite, monkeypatch, capsys):
    def post(url, **kwargs):
        return FakeResponse("no answer tags")

    monkeypatch.setattr(rewrite, "requests",
                        types.SimpleNamespace(post=post, exceptions=requests.exceptions))
    monkeypatch.setattr(rewrite.sys, "argv", ["query_rewrite.py", "--query", "q"])

    with pytest.raises(SystemExit) as info:
        rewrite.main()

    assert info.value.code == 1
    assert "Failed to process response" in capsys.readouterr().err
