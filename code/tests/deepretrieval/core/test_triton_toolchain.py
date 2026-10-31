"""Triton JIT-compiles a C helper at runtime (flash-attn's cross-entropy kernel is on the training path).

It needs a C compiler and the interpreter's Python.h, looked up exactly as triton/runtime/build.py (3.0.0) does.
Found by the GPU smoke test: a venv built on a Debian system Python without python3.10-dev crashed in step 1.
"""
import sys
from pathlib import Path

import pytest

from verl.utils import distributed, toolchain

CODE_DIR = Path(__file__).resolve().parents[3]


def _python_include(monkeypatch, include_dir, scheme="posix_prefix"):
    """Point sysconfig's default-scheme include path (what Triton reads) at include_dir."""
    monkeypatch.setattr(toolchain.sysconfig, "get_default_scheme", lambda: scheme)
    monkeypatch.setattr(toolchain.sysconfig, "get_paths",
                        lambda scheme=None: {"include": str(include_dir) if scheme == "posix_prefix" else "/nonexistent"})


@pytest.fixture
def header_dir(tmp_path):
    (tmp_path / "Python.h").write_text("/* stub */\n")
    return tmp_path


@pytest.fixture
def compiler(tmp_path):
    cc = tmp_path / "bin" / "cc"
    cc.parent.mkdir()
    cc.write_text("#!/bin/sh\n")
    cc.chmod(0o755)
    return cc


def test_missing_python_header_names_the_path_and_the_fix(tmp_path, compiler, monkeypatch):
    _python_include(monkeypatch, tmp_path)
    monkeypatch.setenv("CC", str(compiler))

    with pytest.raises(RuntimeError, match=r"Python\.h not found in .*python-preference = \"only-managed\""):
        toolchain.require_triton_toolchain()


def test_missing_c_compiler_is_reported(header_dir, monkeypatch):
    _python_include(monkeypatch, header_dir)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"no C compiler found .*CC"):
        toolchain.require_triton_toolchain()


def test_cc_naming_a_missing_executable_is_reported(header_dir, monkeypatch):
    _python_include(monkeypatch, header_dir)
    monkeypatch.setenv("CC", "/definitely/not/a/compiler")
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"CC='/definitely/not/a/compiler' is not an executable"):
        toolchain.require_triton_toolchain()


def test_cc_environment_variable_satisfies_the_compiler_requirement(header_dir, compiler, monkeypatch):
    _python_include(monkeypatch, header_dir)
    monkeypatch.setenv("CC", str(compiler))
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: str(compiler) if name == str(compiler) else None)

    assert toolchain.require_triton_toolchain() is None


@pytest.mark.parametrize("found", ["gcc", "clang"])
def test_gcc_or_clang_on_path_satisfies_the_compiler_requirement(header_dir, monkeypatch, found):
    _python_include(monkeypatch, header_dir)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: f"/usr/bin/{name}" if name == found else None)

    assert toolchain.require_triton_toolchain() is None


def test_debian_posix_local_scheme_is_read_as_posix_prefix_like_triton(header_dir, compiler, monkeypatch):
    _python_include(monkeypatch, header_dir, scheme="posix_local")
    monkeypatch.setenv("CC", str(compiler))
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: str(compiler))

    assert toolchain.require_triton_toolchain() is None


def test_the_running_interpreter_has_python_headers():
    """The project's own venv must satisfy the check (uv-managed CPython ships headers)."""
    include = Path(toolchain.python_include_dir())

    assert (include / "Python.h").is_file(), f"{sys.executable} has no Python.h in {include}"


def test_require_cuda_checks_the_triton_toolchain_on_cuda_processes(monkeypatch):
    class _ToolchainChecked(Exception):
        pass

    def fail():
        raise _ToolchainChecked

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(distributed, "require_triton_toolchain", fail)

    with pytest.raises(_ToolchainChecked):
        distributed.require_cuda("worker")


def test_project_pins_uv_managed_python_so_headers_are_always_present():
    import tomli

    pyproject = tomli.loads((CODE_DIR / "pyproject.toml").read_text())

    assert pyproject["tool"]["uv"]["python-preference"] == "only-managed"
