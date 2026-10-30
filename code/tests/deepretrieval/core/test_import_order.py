"""`import utils.java_init` sets JAVA_HOME for pyserini's JVM, so it must run before any pyserini import.

Import sorting (ruff I001) would otherwise move third-party `pyserini` above first-party `utils`.
"""
import ast
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[3]
SKIP_PARTS = {".venv", "third_party", "__pycache__"}


def _module_level_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """(line, module) for imports at module level, including inside top-level try/if blocks."""
    found: list[tuple[int, str]] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append((node.lineno, node.module))
        elif isinstance(node, (ast.Try, ast.If)):
            pending.extend(node.body)
            pending.extend(node.orelse)
            if isinstance(node, ast.Try):
                pending.extend(stmt for handler in node.handlers for stmt in handler.body)
    return found


def _files_importing_java_init() -> list[Path]:
    return sorted(
        path for path in CODE_DIR.rglob("*.py")
        if not SKIP_PARTS & set(path.parts) and "utils.java_init" in path.read_text(errors="ignore")
    )


def test_java_init_importers_are_found():
    assert len(_files_importing_java_init()) >= 10


@pytest.mark.parametrize("path", _files_importing_java_init(), ids=lambda p: str(p.relative_to(CODE_DIR)))
def test_java_init_is_imported_before_pyserini(path):
    imports = _module_level_imports(ast.parse(path.read_text()))
    java_init_lines = [line for line, module in imports if module == "utils.java_init"]
    pyserini_lines = [line for line, module in imports if module.split(".")[0] == "pyserini"]

    assert all(min(java_init_lines, default=0) < line for line in pyserini_lines), (
        f"pyserini imported before utils.java_init in {path}"
    )
