"""src/eval scripts that look up their retrieval index at import honour DEEPRETRIEVAL_DATA_DIR.

Characterisation: with the variable unset and cwd = code/ (how scripts/eval/*.sh run them),
the index each script checks resolves to the same file as the old relative literal did.
"""
import os
from pathlib import Path

import pytest

from tests.deepretrieval.evals._loader import CODE_DIR, load_script

# (script, stubbed search module, old index literal relative to code/)
INDEX_SCRIPTS = [
    *[(f"src/eval/BM25/{task}.py", f"src.Lucene.{task}.search", f"data/local_index_search/{task}/pyserini_index")
      for task in ("fever", "fiqa", "hotpotqa", "msmarco_beir", "nfcorpus", "scifact")],
    ("src/eval/Dense/fever.py", "src.Dense.fever.search",
     "data/local_index_search/fever/dense_index/faiss-flat.beir-v1.0.0-fever.bge-base-en-v1.5.20240107"),
    ("src/eval/Dense/hotpotqa.py", "src.Dense.hotpotqa.search",
     "data/local_index_search/hotpotqa/dense_index/faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107"),
    ("src/eval/Dense/msmarco_beir.py", "src.Dense.msmarco_beir.search",
     "data/local_index_search/msmarco_beir/dense_index/faiss-flat.msmarco-v1-passage.bge-base-en-v1.5.20240107"),
    ("src/eval/Dense/nfcorpus.py", "src.Dense.nfcorpus.search",
     "data/local_index_search/nfcorpus/dense_index/faiss-flat.beir-v1.0.0-nfcorpus.bge-base-en-v1.5.20240107"),
    ("src/eval/Dense/scifact.py", "src.Dense.scifact.search", "data/local_index_search/scifact/dense_index"),
]


def _load_recording_checks(monkeypatch, relpath, search_module):
    checked = []

    def recording_exists(path):
        checked.append(os.path.abspath(path))
        return False  # never build a real searcher

    monkeypatch.setattr(os.path, "exists", recording_exists)
    load_script(monkeypatch, relpath, ("src.Dense", search_module))
    return checked


@pytest.mark.parametrize(("relpath", "search_module", "old_relpath"), INDEX_SCRIPTS)
def test_eval_index_default_matches_the_old_relative_path_from_code_dir(monkeypatch, relpath, search_module,
                                                                       old_relpath):
    monkeypatch.chdir(CODE_DIR)

    checked = _load_recording_checks(monkeypatch, relpath, search_module)

    assert os.path.abspath(old_relpath) in checked


@pytest.mark.parametrize(("relpath", "search_module", "old_relpath"), INDEX_SCRIPTS)
def test_eval_index_follows_the_data_dir_env_var(monkeypatch, tmp_path, relpath, search_module, old_relpath):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path / "volume"))

    checked = _load_recording_checks(monkeypatch, relpath, search_module)

    assert str(tmp_path / "volume" / Path(old_relpath).relative_to("data")) in checked
