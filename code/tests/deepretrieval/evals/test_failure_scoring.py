"""Eval aggregation: a failed or unparseable query scores 0 and stays in the denominator."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from tests.deepretrieval.evals._loader import load_script

LUCENE_STUBS = ("pyserini", "pyserini.search", "pyserini.search.lucene")
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

GOOD = 'Assistant: <think>t</think><answer>{"query": "q"}</answer>'
NO_ANSWER = "Assistant: <think>t</think> no answer tags"


class _Tokenized(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Encodes nothing; decode() returns the canned response for each prompt index."""

    def __init__(self, responses):
        self.responses = responses

    def __call__(self, batch_inputs, **kwargs):
        return _Tokenized(prompt_ids=[int(p) for p in batch_inputs])

    def decode(self, output, **kwargs):
        return self.responses[output]


class FakeModel:
    def to(self, device):
        return self

    def generate(self, prompt_ids, **kwargs):
        return list(prompt_ids)


def _ndcg_lines(text):
    return [float(v) for v in re.findall(r"^NDCG@10: (\S+)$", text, re.MULTILINE)]


@pytest.mark.parametrize(
    ("relpath", "stubs"),
    [("src/eval/BM25/msmarco.py", LUCENE_STUBS), ("src/eval/Dense/msmarco.py", FAISS_STUBS)],
)
def test_msmarco_ndcg_counts_unparseable_response_as_zero(monkeypatch, capsys, relpath, stubs):
    script = load_script(monkeypatch, relpath, stubs)
    frame = pd.DataFrame({
        "prompt": [[{"content": "0"}], [{"content": "1"}]],
        "label": [np.array(["d1"]), np.array(["d2"])],
    })
    monkeypatch.setattr(script.pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(script, "retriver_items", lambda query, top_k, mode: ["d1"])

    script.evaluate_model(FakeModel(), FakeTokenizer([GOOD, NO_ANSWER]), "test.parquet", "cpu",
                          "m", "out", batch_size=2)

    assert _ndcg_lines(capsys.readouterr().out) == [pytest.approx(0.5)]


@pytest.fixture
def evidence_script(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # process_generations writes results/ under cwd
    return load_script(monkeypatch, "src/eval/baselines/answer_filtered_test.py", EVIDENCE_STUBS)


def _write_inputs(tmp_path, generations):
    frame = pd.DataFrame({
        "input": [f"raw {i}" for i in range(len(generations))],
        "label": [np.array(["ans"]) for _ in generations],
    })
    data_path = tmp_path / "test.parquet"
    frame.to_parquet(data_path)
    generations_path = tmp_path / "generations.json"
    generations_path.write_text(json.dumps(generations))
    return str(generations_path), str(data_path)


HIT = {f"recall@{k}": 1 for k in (1, 5, 10, 20, 50, 100)}
NO_INJECTION = '```json\n{"has_answer": "false", "cannot_be_derived": "false"}\n```'


def test_answer_filter_counts_unparseable_generation_as_zero(evidence_script, tmp_path, monkeypatch):
    generations_path, data_path = _write_inputs(tmp_path, ['{"query": "q"}', "not json at all"])
    monkeypatch.setattr(evidence_script, "evaluate_query", lambda query, target: dict(HIT))
    monkeypatch.setattr(evidence_script, "get_if_answer_span_in_query_batch",
                        lambda raw, queries, targets, batch_size: [NO_INJECTION] * len(queries))

    results = evidence_script.process_generations("nq", "m", generations_path, data_path)

    assert results["original_metrics"]["recall@1"] == pytest.approx(0.5)
    assert results["cleaned_metrics"]["recall@1"] == pytest.approx(0.5)


def test_answer_filter_counts_unparseable_injection_check_as_zero_after_cleaning(
        evidence_script, tmp_path, monkeypatch):
    generations_path, data_path = _write_inputs(tmp_path, ['{"query": "q"}', '{"query": "r"}'])
    monkeypatch.setattr(evidence_script, "evaluate_query", lambda query, target: dict(HIT))
    monkeypatch.setattr(evidence_script, "get_if_answer_span_in_query_batch",
                        lambda raw, queries, targets, batch_size: [NO_INJECTION, "no json here"])

    results = evidence_script.process_generations("nq", "m", generations_path, data_path)

    assert results["original_metrics"]["recall@1"] == pytest.approx(1.0)
    assert results["cleaned_metrics"]["recall@1"] == pytest.approx(0.5)


def test_answer_filter_scores_each_query_once_when_injection_handling_fails(
        evidence_script, tmp_path, monkeypatch):
    generations_path, data_path = _write_inputs(tmp_path, ['{"query": "q"}', '{"query": "r"}'])
    monkeypatch.setattr(evidence_script, "evaluate_query", lambda query, target: dict(HIT))
    monkeypatch.setattr(evidence_script, "get_if_answer_span_in_query_batch",
                        lambda raw, queries, targets, batch_size: [NO_INJECTION])  # one short

    results = evidence_script.process_generations("nq", "m", generations_path, data_path)

    # Query 1's original score (a hit) is kept; only its cleaned score is a failure.
    assert results["original_metrics"]["recall@1"] == pytest.approx(1.0)
    assert results["cleaned_metrics"]["recall@1"] == pytest.approx(0.5)
