"""Fixtures for the core (verl/) audit-fix tests.

flash-attn and vllm are CUDA-only and absent on CPU dev boxes; modules that import them at
top level are imported here against stub modules installed with monkeypatch.
"""
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[3]


def _not_available(*args, **kwargs):
    raise AssertionError("flash-attn stub called; the test should not reach real flash-attn code")


@pytest.fixture
def fresh_import(monkeypatch):
    """Import a module from scratch, dropping it (and the listed parents) from sys.modules first.

    monkeypatch restores the previous sys.modules entries on teardown.
    """

    def _import(name, *also_drop):
        for mod in (name, *also_drop):
            monkeypatch.delitem(sys.modules, mod, raising=False)
        return importlib.import_module(name)

    return _import


@pytest.fixture
def stub_flash_attn(monkeypatch):
    """Install a fake ``flash_attn.bert_padding`` exposing the names verl imports."""
    flash_attn = types.ModuleType("flash_attn")
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    for attr in ("pad_input", "unpad_input", "rearrange", "index_first_axis"):
        setattr(bert_padding, attr, _not_available)
    vars(flash_attn)["bert_padding"] = bert_padding
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.bert_padding", bert_padding)
    return bert_padding


@pytest.fixture
def no_flash_attn(monkeypatch):
    """Make every ``import flash_attn...`` raise ImportError, as on a CPU-only install."""
    for mod in ("flash_attn", "flash_attn.bert_padding"):
        monkeypatch.setitem(sys.modules, mod, None)


@pytest.fixture
def no_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def run_ruff(rules, path):
    """Run ruff with the given rule codes on one file (relative to code/); return its findings text."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--quiet", "--no-cache", "--select", rules, "--output-format", "concise", path],
        cwd=CODE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


@pytest.fixture
def ruff_findings():
    return run_ruff
