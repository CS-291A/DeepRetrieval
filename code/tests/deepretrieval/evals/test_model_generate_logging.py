"""Baseline query-generation scripts log every failed API attempt and every parse fallback (audit LOG007).

Control flow is unchanged: the same number of retries, the same "" / raw-text fallbacks.
"""
import json
import logging
import runpy
import sys

import pandas as pd
import pytest

from tests.deepretrieval.evals._loader import CODE_DIR, load_script, stub_modules

BM25_GEN = "src/eval/BM25/baselines/model_generate"


def _prompt(text):
    return [{"role": "user", "content": f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"}]


def _write_parquet(path, columns):
    pd.DataFrame(columns).to_parquet(path)
    return path


def _run_main(monkeypatch, relpath, argv):
    monkeypatch.setattr(sys, "path", [str(CODE_DIR), *sys.path])
    monkeypatch.setattr(sys, "argv", [str(CODE_DIR / relpath), *argv])
    runpy.run_path(str(CODE_DIR / relpath), run_name="__main__")


class _FlakyAPI:
    """Raises for the first `failures` calls, then returns `answer`."""

    def __init__(self, failures, answer="rewritten query"):
        self.failures = failures
        self.answer = answer
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError(f"throttled #{self.calls}")
        return self.answer


@pytest.fixture
def api_inputs(tmp_path):
    data = _write_parquet(tmp_path / "test.parquet", {"prompt": [_prompt("find papers")], "target": ["d1"], "qid": ["q1"]})
    return ["--data_path", str(data), "--save_dir", str(tmp_path / "out")]


def _api_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize(
    ("script", "sdk_module", "fn_name", "model_name"),
    [
        ("claude.py", "src.utils.claude_aws", "chat_sonnet", "claude-3.5"),
        ("gpt.py", "src.utils.gpt_azure", "gpt_chat_4omini", "gpt-4o-mini"),
    ],
)
def test_each_failed_api_attempt_is_logged_with_attempt_number(
    monkeypatch, caplog, tmp_path, api_inputs, script, sdk_module, fn_name, model_name
):
    flaky = _FlakyAPI(failures=2)
    stubs = stub_modules(monkeypatch, ["src", "src.utils", sdk_module])
    setattr(stubs[sdk_module], fn_name, flaky)

    with caplog.at_level(logging.INFO):
        _run_main(monkeypatch, f"{BM25_GEN}/{script}", [*api_inputs, "--model_name", model_name])

    warnings = _api_warnings(caplog)
    assert [r.args[-1] for r in warnings] == [1, 2]
    assert all(r.exc_info is not None and r.exc_info[0] is ConnectionError for r in warnings)
    saved = json.loads((tmp_path / "out" / f"{model_name}_scifact.json").read_text())
    assert saved == {"q1": {"generated_text": "rewritten query", "target": "d1"}}


def test_exhausted_retries_log_ten_attempts_and_record_empty_query(monkeypatch, caplog, tmp_path, api_inputs):
    flaky = _FlakyAPI(failures=100)
    stubs = stub_modules(monkeypatch, ["src", "src.utils", "src.utils.claude_aws"])
    stubs["src.utils.claude_aws"].chat_sonnet = flaky

    with caplog.at_level(logging.INFO):
        _run_main(monkeypatch, f"{BM25_GEN}/claude.py", api_inputs)

    assert flaky.calls == 10
    assert [r.args[-1] for r in _api_warnings(caplog)] == list(range(1, 11))
    saved = json.loads((tmp_path / "out" / "claude-3.5_scifact.json").read_text())
    assert saved["q1"]["generated_text"] == ""


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, decoded):
        self.decoded = decoded

    def __call__(self, texts, **kwargs):
        return _FakeBatch()

    def decode(self, output, skip_special_tokens):
        return self.decoded[output]


class _FakeModel:
    def __init__(self, n):
        self.n = n

    def to(self, device):
        return self

    def generate(self, **kwargs):
        return list(range(self.n))


@pytest.mark.parametrize(
    "relpath",
    [
        f"{BM25_GEN}/model_generate.py",
        "src/eval/Dense/baselines/model_generate/model_generate.py",
        "src/eval/Dense/baselines/model_generate/msmarco_subset/model_generate.py",
    ],
)
def test_local_model_json_fallback_is_logged(monkeypatch, caplog, tmp_path, relpath):
    script = load_script(monkeypatch, relpath)
    data = _write_parquet(tmp_path / "t.parquet", {"prompt": [_prompt("q")], "target": ["d1"], "label": ["d1"], "qid": ["q1"]})
    tokenizer = _FakeTokenizer(["sys\nAssistant: <answer>plain words</answer>"])

    with caplog.at_level(logging.DEBUG):
        script.evaluate_model(_FakeModel(1), tokenizer, str(data), "cpu", "m", str(tmp_path))

    saved = json.loads((tmp_path / "m.json").read_text())
    assert [v["generated_text"] for v in saved.values()] == ["plain words"]
    [record] = [r for r in caplog.records if r.name == script.__name__]
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None


def test_local_model_column_and_case_fallbacks_are_logged(monkeypatch, caplog, tmp_path):
    script = load_script(monkeypatch, f"{BM25_GEN}/model_generate.py")
    data = _write_parquet(tmp_path / "t.parquet", {"prompt": [_prompt("q")], "label": ["d1"]})
    tokenizer = _FakeTokenizer(['sys\nassistant: <answer>{"query": "q x"}</answer>'])

    with caplog.at_level(logging.DEBUG):
        script.evaluate_model(_FakeModel(1), tokenizer, str(data), "cpu", "m", str(tmp_path))

    saved = json.loads((tmp_path / "m.json").read_text())
    assert saved == {"0": {"generated_text": "q x", "target": "d1"}}
    messages = [r.getMessage() for r in caplog.records if r.name == script.__name__]
    assert len(messages) == 3
    assert "'label'" in messages[0]
    assert "row index" in messages[1]
    assert "Assistant:" in messages[2]


def _post_process_inputs(tmp_path, results_dir, filename):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / results_dir).mkdir(parents=True)
    data = {
        "k_json": {"generated_text": '<answer>{"query": "good"}</answer>', "target": "t"},
        "k_raw": {"generated_text": "no tags at all", "target": "t"},
    }
    (tmp_path / results_dir / filename).write_text(json.dumps(data))
    return work


@pytest.mark.parametrize(
    ("relpath", "results_dir", "infile", "outfile"),
    [
        (f"{BM25_GEN}/post_process.py", "results", "gpt-35_scifact.json", "gpt-35_post_scifact.json"),
        ("src/eval/Dense/baselines/model_generate/post_process/claude.py", "results_dense",
         "claude-3.5_msmarco_beir.json", "claude-3.5_post_msmarco_beir.json"),
        ("src/eval/Dense/baselines/model_generate_no_reasoning/post_process/gpt.py", "results_dense/no_reason",
         "gpt-4o_msmarco_beir.json", "gpt-4o_post_msmarco_beir.json"),
    ],
)
def test_post_process_raw_text_fallback_is_logged(monkeypatch, caplog, tmp_path, relpath, results_dir, infile, outfile):
    work = _post_process_inputs(tmp_path, results_dir, infile)
    monkeypatch.chdir(work)

    with caplog.at_level(logging.INFO):
        load_script(monkeypatch, relpath)

    saved = json.loads((tmp_path / results_dir / outfile).read_text())
    assert saved["k_json"]["generated_text"] == "good"
    assert saved["k_raw"]["generated_text"] == "no tags at all"
    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.args == ("k_raw",)
    assert record.exc_info is not None
