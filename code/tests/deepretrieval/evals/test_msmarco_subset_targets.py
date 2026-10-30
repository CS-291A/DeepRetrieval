"""{BM25,Dense}/baselines/search/msmarco_subset.py parse the stored `target` as a literal only."""
import json

import pytest

from tests.deepretrieval.evals._loader import run_script_as_main

SCRIPTS = [
    ("src/eval/BM25/baselines/search/msmarco_subset.py",
     ("pyserini", "pyserini.search", "pyserini.search.lucene"),
     "results/claude-haiku_post_msmarco_tech.json"),
    ("src/eval/Dense/baselines/search/msmarco_subset.py",
     ("pyserini", "pyserini.search", "pyserini.search.faiss"),
     "results_dense/no_reason/gpt-35_post_msmarco_tech.json"),
]


@pytest.mark.parametrize(("relpath", "stubs", "results_file"), SCRIPTS)
def test_target_that_is_code_is_rejected_not_executed(tmp_path, monkeypatch, relpath, stubs,
                                                      results_file):
    # The scripts read ../<results_file>; work one level below tmp_path.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    target_file = tmp_path / results_file
    target_file.parent.mkdir(parents=True)
    payload = "[__import__('pathlib').Path('pwned').touch()]"
    target_file.write_text(json.dumps({"0": {"generated_text": "q", "target": payload}}))

    with pytest.raises(ValueError, match="malformed node or string"):
        run_script_as_main(monkeypatch, relpath, stubs)

    assert not (work / "pwned").exists()
