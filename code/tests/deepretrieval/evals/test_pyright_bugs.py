"""Runtime bugs in src/ that pyright reported (unbound names, wrong call signatures, dead fallbacks)."""
import json
import runpy
import sys

import pandas as pd
import pytest

from tests.deepretrieval.evals._loader import CODE_DIR, load_script, stub_modules
from verl.utils import credentials

FAISS_STUBS = ("pyserini", "pyserini.search", "pyserini.search.faiss")
EVIDENCE_STUBS = (
    "src.utils.claude_api",
    "utils.java_init",
    "pyserini",
    "pyserini.search",
    "pyserini.search.lucene",
    "pyserini.eval",
    "pyserini.eval.evaluate_dpr_retrieval",
)
AZURE_STUBS = (
    "openai",
    "langchain_openai",
    "langchain_community",
    "langchain_community.callbacks",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
)
LLM_CLIENT_STUBS = ("src.utils.claude_api", "src.utils.gpt_azure")


def _run_main(monkeypatch, relpath, stubs, argv=()):
    """Run a script as __main__ with `argv` and the named modules stubbed; return the stubs."""
    path = CODE_DIR / relpath
    assert path.is_file(), f"no such script: {path}"
    installed = stub_modules(monkeypatch, stubs)
    monkeypatch.setattr(sys, "path", [str(CODE_DIR), *sys.path])
    monkeypatch.setattr(sys, "argv", [str(path), *argv])
    runpy.run_path(str(path), run_name="__main__")
    return installed


# --- PubMed evals: a missing key file left `search_api` unbound --------------------------------

PUBMED_MAIN_SCRIPTS = (
    ("src/eval/SearchEngine/eval_pubmed.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_no_reason.py", ()),
    ("src/eval/SearchEngine/eval_ctgov.py", ("--api", "pubmed")),
    ("src/eval/SearchEngine/eval_pubmed_base.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_reason.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_no_reason_gpt.py", ()),
)


@pytest.mark.parametrize(("relpath", "argv"), PUBMED_MAIN_SCRIPTS)
def test_pubmed_eval_without_key_file_names_the_missing_key(monkeypatch, tmp_path, relpath, argv):
    monkeypatch.chdir(tmp_path)  # no verl/utils/reward_score/apis/pubmed_api.key here
    monkeypatch.setattr(credentials, "DEFAULT_PUBMED_API_KEY_PATH", tmp_path / "pubmed_api.key")  # nor in the package
    script = load_script(monkeypatch, relpath, LLM_CLIENT_STUBS)
    monkeypatch.setattr(sys, "argv", [relpath, *argv])
    # Scripts that load a model must fail before doing so; the others don't define load_model.
    monkeypatch.setattr(script, "load_model", lambda path: (object(), object()), raising=False)

    with pytest.raises(FileNotFoundError, match="pubmed_api.key"):
        script.main()


# --- Dense msmarco evals: the prebuilt fallback could never build a searcher -------------------

@pytest.mark.parametrize("relpath", ["src/eval/Dense/msmarco.py", "src/eval/Dense/msmarco_sub.py"])
def test_dense_msmarco_missing_index_raises_instead_of_dead_prebuilt_fallback(monkeypatch, relpath):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # msmarco_sub.py overwrites it on import
    script = load_script(monkeypatch, relpath, FAISS_STUBS)
    monkeypatch.setattr(script, "index_dir", "/nonexistent/dense-index")

    with pytest.raises(FileNotFoundError, match="/nonexistent/dense-index"):
        script.get_searcher()
    script.FaissSearcher.from_prebuilt_index.assert_not_called()


# --- model_generate baselines: failed generations reused the previous prompt's output ----------

MODEL_GENERATE = (
    # (script, client module, function the default --model_name calls)
    ("src/eval/BM25/baselines/model_generate/claude.py", "src.utils.claude_aws", "chat_sonnet"),
    ("src/eval/BM25/baselines/model_generate/gpt.py", "src.utils.gpt_azure", "gpt_chat_4o"),
    ("src/eval/Dense/baselines/model_generate/claude.py", "src.utils.claude_aws", "chat_sonnet"),
    ("src/eval/Dense/baselines/model_generate/gpt-4o.py", "src.utils.gpt_azure", "gpt_chat_35_msg"),
    ("src/eval/Dense/baselines/model_generate/msmarco_subset/claude.py", "src.utils.claude_aws", "chat_sonnet"),
    ("src/eval/Dense/baselines/model_generate/msmarco_subset/gpt-4o.py", "src.utils.gpt_azure", "gpt_chat_4o"),
    ("src/eval/Dense/baselines/model_generate_no_reasoning/claude.py", "src.utils.claude_aws", "chat_sonnet"),
    ("src/eval/Dense/baselines/model_generate_no_reasoning/gpt-4o.py", "src.utils.gpt_azure", "gpt_chat_35_msg"),
    ("src/eval/Dense/baselines/model_generate_no_reasoning/msmarco_subset/claude.py", "src.utils.claude_aws",
     "chat_sonnet"),
    ("src/eval/Dense/baselines/model_generate_no_reasoning/msmarco_subset/gpt-4o.py", "src.utils.gpt_azure",
     "gpt_chat_4o"),
)


def _write_prompts(tmp_path, n):
    frame = pd.DataFrame({
        "prompt": [[{"content": f"<|im_start|>user prompt {i}<|im_start|>assistant"}] for i in range(n)],
        "target": [f"t{i}" for i in range(n)],
        "label": [f"t{i}" for i in range(n)],  # the msmarco_subset scripts read `label`
        "qid": [f"q{i}" for i in range(n)],
    })
    data_path = tmp_path / "test.parquet"
    frame.to_parquet(data_path)
    return data_path


def _install_client(monkeypatch, client_module, func_name, func):
    stubs = stub_modules(monkeypatch, (client_module,))
    # Real signatures, so a call with the wrong arguments fails the way the real client would.
    stubs[client_module].gpt_chat_35 = lambda prompt, query_dict: "via prompt template"
    setattr(stubs[client_module], func_name, func)


@pytest.mark.parametrize(("relpath", "client_module", "func_name"), MODEL_GENERATE)
def test_model_generate_records_empty_text_when_every_attempt_fails(
        monkeypatch, tmp_path, relpath, client_module, func_name):
    monkeypatch.chdir(tmp_path)
    data_path = _write_prompts(tmp_path, 2)

    def answer_only_the_first_prompt(prompt, max_tokens=None):
        if "prompt 0" in prompt:  # audit: ignore[BRANCHING] fake API: answers prompt 0, fails the rest
            return "answer for prompt 0"
        raise RuntimeError("API down")

    _install_client(monkeypatch, client_module, func_name, answer_only_the_first_prompt)
    _run_main(monkeypatch, relpath, (), ["--data_path", str(data_path), "--save_dir", str(tmp_path / "out"),
                                         "--dataset", "d"])

    [result_file] = (tmp_path / "out").glob("*_d.json")
    # Keyed by the qid column, or by row number in the msmarco_subset scripts; rows stay in order.
    first, second = json.loads(result_file.read_text()).values()
    assert first["generated_text"] == "answer for prompt 0"
    assert second["generated_text"] == ""


@pytest.mark.parametrize(("relpath", "client_module", "func_name"), MODEL_GENERATE)
def test_model_generate_rejects_unknown_model_name(monkeypatch, tmp_path, relpath, client_module, func_name):
    monkeypatch.chdir(tmp_path)
    data_path = _write_prompts(tmp_path, 1)
    _install_client(monkeypatch, client_module, func_name, lambda prompt, max_tokens=None: "x")

    with pytest.raises(SystemExit) as excinfo:  # argparse rejects it before any API call
        _run_main(monkeypatch, relpath, (), ["--data_path", str(data_path), "--save_dir", str(tmp_path / "out"),
                                             "--model_name", "no-such-model"])
    assert excinfo.value.code == 2


# --- gpt_azure.py demo called gpt_chat_35 without its query_dict --------------------------------

def test_gpt_azure_demo_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_TYPE", "test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test")
    monkeypatch.setenv("OPENAI_API_VERSION", "test")
    stubs = stub_modules(monkeypatch, AZURE_STUBS)
    stubs["langchain_openai"].AzureChatOpenAI.return_value.invoke.return_value.content = "J'aime programmer."

    _run_main(monkeypatch, "src/utils/gpt_azure.py", ())

    assert "J'aime programmer." in capsys.readouterr().out


# --- answer_filtered_test.py: single-query wrapper passed too few arguments ---------------------

def test_answer_span_single_query_wrapper_forwards_all_batch_arguments(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    script = load_script(monkeypatch, "src/eval/baselines/answer_filtered_test.py", EVIDENCE_STUBS)
    calls = []

    def fake_batch(original_queries, queries, answer_candidates_list, batch_size=10):
        calls.append((original_queries, queries, answer_candidates_list, batch_size))
        return ["response"]

    monkeypatch.setattr(script, "get_if_answer_span_in_query_batch", fake_batch)

    assert script.get_if_answer_span_in_query("original", "rewritten", ["ans"]) == "response"
    assert calls == [(["original"], ["rewritten"], [["ans"]], 1)]


# --- claude_aws.py: Claude 3.x baselines on Bedrock need anthropic 1.x --------------------------
# anthropic>=2 sends API version 2026-10-21, which Bedrock accepts only for Claude 4.7+ and which
# rejects `temperature` (even via extra_body). 1.x still sends bedrock-2023-05-31, where the
# extra_body temperature is accepted.

@pytest.mark.parametrize(("func_name", "model"), [
    ("chat_haiku", "anthropic.claude-3-haiku-20240307-v1:0"),
    ("chat_sonnet", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
])
def test_claude_aws_sends_temperature_zero_with_the_installed_sdk(monkeypatch, func_name, model):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")  # the SDK ships its own httpx fork
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={  # audit: ignore[NETWORK] response object for MockTransport, no I/O
            "id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "a haiku"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    client = anthropic.AnthropicBedrock(
        aws_access_key="test", aws_secret_key="test", aws_region="us-west-2", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),  # audit: ignore[NETWORK] MockTransport never opens a socket
    )
    claude_aws = load_script(monkeypatch, "src/utils/claude_aws.py",
                             ("langchain_anthropic", "langchain_core", "langchain_core.prompts"))
    monkeypatch.setattr(claude_aws, "_client", client)

    assert getattr(claude_aws, func_name)("Write a haiku.") == "a haiku"
    assert len(sent) == 1
    assert sent[0]["anthropic_version"] == "bedrock-2023-05-31"
    assert sent[0]["temperature"] == 0
    assert sent[0]["max_tokens"] == 1024
    assert sent[0]["messages"] == [{"role": "user", "content": "Write a haiku."}]
