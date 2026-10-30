"""Shared pytest configuration.

Most of the upstream verl suite needs CUDA, vllm or flash-attn. On machines without them
(e.g. a macOS dev box) those tests are skipped or not collected instead of erroring.
"""
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS_DIR = Path(__file__).parent

# Modules missing on CPU-only installs: test files that import them at module level
# would raise during collection, so don't collect them at all.
collect_ignore_glob = []
if importlib.util.find_spec("flash_attn") is None:
    collect_ignore_glob += ["model/*", "gpu_utility/test_torch_functional.py"]
if importlib.util.find_spec("vllm") is None:
    collect_ignore_glob += ["rollout/*"]

# Directories whose tests place work on CUDA devices (Ray GPU placement groups wait forever without GPUs).
_GPU_DIRS = ("ray", "gpu_utility", "model", "rollout", "e2e")
# Directories whose tests read preprocessed parquet files from ~/verl-data.
_DATA_DIRS = (("verl", "utils", "dataset"),)
_VERL_DATA = Path(os.path.expanduser("~/verl-data"))


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    # The dataset tests mkdir their own folders, so check for actual parquet files.
    has_data = _VERL_DATA.is_dir() and any(_VERL_DATA.rglob("*.parquet"))
    for item in items:
        parts = item.path.relative_to(_TESTS_DIR).parts
        if parts[0] in _GPU_DIRS:
            item.add_marker(pytest.mark.gpu)
            if not has_cuda:
                item.add_marker(pytest.mark.skip(reason="needs CUDA"))
        if any(parts[:len(d)] == d for d in _DATA_DIRS):
            item.add_marker(pytest.mark.data)
            if not has_data:
                item.add_marker(pytest.mark.skip(reason=f"needs preprocessed data in {_VERL_DATA}"))


@pytest.fixture(autouse=True)
def _seed_rngs():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
