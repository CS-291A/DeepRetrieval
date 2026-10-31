"""Reward modules resolve their retrieval index through verl.utils.paths.data_path at import.

Characterisation: with DEEPRETRIEVAL_DATA_DIR unset, each module checks the same directory its
old cwd-relative literal named when run from code/; with it set, the index is found (and the
search backend constructed) under the configured directory.
"""
import logging
import os
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[3]

# (module, the old index literal, relative to code/)
REWARD_INDEXES = [
    ("verl.utils.reward_score.fever", "data/local_index_search/fever/pyserini_index"),
    ("verl.utils.reward_score.fiqa", "data/local_index_search/fiqa/pyserini_index"),
    ("verl.utils.reward_score.hotpotqa", "data/local_index_search/hotpotqa/pyserini_index"),
    ("verl.utils.reward_score.msmarco_beir", "data/local_index_search/msmarco_beir/pyserini_index"),
    ("verl.utils.reward_score.nfcorpus", "data/local_index_search/nfcorpus/pyserini_index"),
    ("verl.utils.reward_score.scifact", "data/local_index_search/scifact/pyserini_index"),
    ("verl.utils.reward_score_dense.fever",
     "data/local_index_search/fever/dense_index/faiss-flat.beir-v1.0.0-fever.bge-base-en-v1.5.20240107"),
    ("verl.utils.reward_score_dense.hotpotqa",
     "data/local_index_search/hotpotqa/dense_index/faiss-flat.beir-v1.0.0-hotpotqa.bge-base-en-v1.5.20240107"),
    ("verl.utils.reward_score_dense.msmarco_beir",
     "data/local_index_search/msmarco_beir/dense_index/faiss-flat.msmarco-v1-passage.bge-base-en-v1.5.20240107"),
    ("verl.utils.reward_score_dense.nfcorpus",
     "data/local_index_search/nfcorpus/dense_index/faiss-flat.beir-v1.0.0-nfcorpus.bge-base-en-v1.5.20240107"),
    ("verl.utils.reward_score_dense.scifact", "data/local_index_search/scifact/dense_index"),
]


class RecordingBackend:
    """Stands in for a search class; records the arguments it was constructed with."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append((args, kwargs))


@pytest.fixture
def recording_backend(monkeypatch):
    monkeypatch.setattr(RecordingBackend, "calls", [])
    return RecordingBackend


def _path_arguments(call):
    """Filesystem arguments only: absolute or data-relative strings (not HF model ids like BAAI/bge-...)."""
    args, kwargs = call
    return [str(value) for value in (*args, *kwargs.values())
            if os.path.isabs(str(value)) or str(value).startswith("data/")]


@pytest.mark.parametrize(("module_name", "old_relpath"), REWARD_INDEXES)
def test_reward_index_default_is_the_old_path_under_code_dir(load_reward_module, monkeypatch, module_name,
                                                            old_relpath):
    checked = []

    def recording_exists(path):
        checked.append(str(path))
        return False  # never open a developer's real index

    monkeypatch.setattr(os.path, "exists", recording_exists)

    load_reward_module(module_name, isolate_data_dir=False)  # chdirs to an empty tmp dir: the default must not follow the cwd

    assert str(CODE_DIR / old_relpath) in checked


@pytest.mark.parametrize(("module_name", "old_relpath"), REWARD_INDEXES)
def test_reward_index_follows_the_data_dir_env_var(load_reward_module, recording_backend, monkeypatch, tmp_path,
                                                  module_name, old_relpath):
    index = tmp_path / "volume" / old_relpath.removeprefix("data/")
    index.mkdir(parents=True)
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path / "volume"))

    module = load_reward_module(module_name, search_backend=recording_backend)

    assert isinstance(module.search_system, recording_backend)
    assert len(recording_backend.calls) == 1
    path_args = _path_arguments(recording_backend.calls[0])
    assert path_args, f"backend got no path: {recording_backend.calls[0]}"
    assert all(p.startswith(str(index)) for p in path_args)


@pytest.mark.parametrize(("module_name", "old_relpath"), REWARD_INDEXES)
def test_missing_reward_index_warning_names_path_and_env_var(load_reward_module, monkeypatch, tmp_path, caplog,
                                                            module_name, old_relpath):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path / "empty"))

    with caplog.at_level(logging.WARNING, logger=module_name):
        module = load_reward_module(module_name)

    assert module.search_system is None
    assert str(tmp_path / "empty" / old_relpath.removeprefix("data/")) in caplog.text
    assert "DEEPRETRIEVAL_DATA_DIR" in caplog.text
