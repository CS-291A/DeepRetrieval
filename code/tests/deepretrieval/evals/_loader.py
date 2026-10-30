"""Load standalone `src/` scripts in tests without their heavy or unavailable dependencies.

The scripts under `src/` import SDKs (openai, boto3, langchain, pyserini, vllm, ...) at module
level and are not a package. `load_script` executes one from its file path after replacing the
named modules in `sys.modules` with permissive stubs; `monkeypatch` undoes every change.
"""
import ast
import importlib.util
import re
import runpy
import sys
import types
from pathlib import Path
from unittest import mock

CODE_DIR = Path(__file__).resolve().parents[3]


class StubModule(types.ModuleType):
    """A module whose every public attribute is a MagicMock, so `from x import Anything` works."""

    __path__ = []  # importable as a package, so dotted children resolve

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def stub_modules(monkeypatch, names):
    """Install a StubModule for each dotted name. Parents must be listed explicitly."""
    stubs = {}
    for name in names:
        stub = StubModule(name)
        monkeypatch.setitem(sys.modules, name, stub)
        stubs[name] = stub
        parent, _, child = name.rpartition(".")
        if parent in stubs:
            setattr(stubs[parent], child, stub)
    return stubs


def load_script(monkeypatch, relpath, stubs=()):
    """Execute `CODE_DIR/relpath` as a fresh module and return it."""
    path = CODE_DIR / relpath
    assert path.is_file(), f"no such script: {path}"
    stub_modules(monkeypatch, stubs)
    # Scripts insert into sys.path and parse sys.argv at import; keep both contained.
    monkeypatch.setattr(sys, "path", [str(CODE_DIR), *sys.path])
    monkeypatch.setattr(sys, "argv", [str(path)])
    name = "_script_" + re.sub(r"\W", "_", relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"no import spec for {path}"
    assert spec.loader is not None, f"no loader for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script_as_main(monkeypatch, relpath, stubs=()):
    """Run `CODE_DIR/relpath` with __name__ == "__main__" and return its globals."""
    path = CODE_DIR / relpath
    assert path.is_file(), f"no such script: {path}"
    stub_modules(monkeypatch, stubs)
    monkeypatch.setattr(sys, "path", [str(CODE_DIR), *sys.path])
    monkeypatch.setattr(sys, "argv", [str(path)])
    return runpy.run_path(str(path), run_name="__main__")


def load_function(relpath, func_name, namespace):
    """Compile only `func_name` from a script, with `namespace` as its globals.

    For pure helpers in scripts whose module body does work on import (parses args, opens
    files, starts a JVM). The function body is the real source, so a fix there is exercised.
    """
    source = (CODE_DIR / relpath).read_text()
    tree = ast.parse(source)
    defs: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    assert len(defs) == 1, f"expected one top-level def {func_name} in {relpath}, found {len(defs)}"
    module = ast.Module(body=defs, type_ignores=[])
    code = compile(module, str(CODE_DIR / relpath), "exec")
    scope = dict(namespace)
    exec(code, scope)  # noqa: S102 - executes repository source, not external input
    return scope[func_name]
