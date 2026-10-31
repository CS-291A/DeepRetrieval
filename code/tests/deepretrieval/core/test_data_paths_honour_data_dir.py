"""Every dataset / index path honours DEEPRETRIEVAL_DATA_DIR (verl.utils.paths.data_path).

The guard scans the Python sources for string literals that hardcode a cwd-relative data
folder (`data/...`, `./data`, `code/data/...`). The characterisation tests import the modules
that resolve paths at import time and check both settings: unset -> the same files the old
cwd-relative paths named with cwd = code/; set -> under the configured directory.
"""
import ast
import os
import re
from pathlib import Path

import pytest

from tests.deepretrieval.evals._loader import load_script

CODE_DIR = Path(__file__).resolve().parents[3]
SCANNED = ("verl", "src", "data_preprocess", "download_datasets.py", "upload_datasets.py")
EXCLUDED_PARTS = {"third_party", "__pycache__"}
PATHS_MODULE = CODE_DIR / "verl" / "utils" / "paths.py"  # defines the stored `data/` reference format
HARDCODED_DATA = re.compile(r"^(\./|code/)data(/|$)|^data/")


def _python_files():
    for entry in SCANNED:
        root = CODE_DIR / entry
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        yield from (f for f in files if not EXCLUDED_PARTS.intersection(f.parts) and f != PATHS_MODULE)


def _hardcoded_literals(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):  # f"data/{x}": the first literal chunk carries the prefix
            first = node.values[0] if node.values else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and HARDCODED_DATA.match(first.value):
                yield node.lineno, first.value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and HARDCODED_DATA.match(node.value):
            yield node.lineno, node.value


def test_no_python_source_hardcodes_a_relative_data_path():
    offenders = [f"{path.relative_to(CODE_DIR)}:{line}: {value!r}"
                 for path in _python_files() for line, value in _hardcoded_literals(path)]

    assert offenders == [], "use verl.utils.paths.data_path(...) instead:\n" + "\n".join(offenders)


# --- SQL rewards: db_path stored in the parquet ----------------------------------------------

def test_sql_db_path_stored_as_data_relative_is_resolved_under_the_data_dir(monkeypatch, tmp_path):
    from verl.trainer import reward_registry

    db = tmp_path / "raw_data" / "bird" / "dev" / "db.sqlite"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path))
    sample = {"extra_info": {"db_path": "data/raw_data/bird/dev/db.sqlite"}}

    kwargs = reward_registry.sql_kwargs(sample, "bird_test", reward_registry.SearchApis(pubmed=None, ctgov=None))

    assert kwargs == {"data_source": "bird_test", "db_path": str(db)}


def test_sql_db_path_default_matches_the_old_cwd_relative_file(monkeypatch):
    from verl.trainer import reward_registry

    monkeypatch.chdir(CODE_DIR)
    stored = "data/raw_data/bird/dev/db.sqlite"
    monkeypatch.setattr(Path, "exists", lambda self: True)

    kwargs = reward_registry.sql_kwargs({"extra_info": {"db_path": stored}}, "bird_test",
                                        reward_registry.SearchApis(pubmed=None, ctgov=None))

    assert kwargs["db_path"] == os.path.abspath(stored)


def test_sql_missing_database_raises_naming_path_and_env_var(monkeypatch, tmp_path):
    from verl.trainer import reward_registry

    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="DEEPRETRIEVAL_DATA_DIR") as excinfo:
        reward_registry.sql_kwargs({"extra_info": {"db_path": "data/raw_data/bird/missing.sqlite"}}, "bird_test",
                                   reward_registry.SearchApis(pubmed=None, ctgov=None))

    assert str(tmp_path / "raw_data" / "bird" / "missing.sqlite") in str(excinfo.value)


# --- download / upload defaults --------------------------------------------------------------

@pytest.mark.parametrize(("relpath", "option", "argv_extra"), [
    ("download_datasets.py", "output_dir", []),
    ("upload_datasets.py", "input_dir", ["--token", "t"]),
])
def test_dataset_transfer_default_dir_follows_the_data_dir(monkeypatch, tmp_path, relpath, option, argv_extra):
    import sys

    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path / "volume"))
    script = load_script(monkeypatch, relpath, ("huggingface_hub",))
    monkeypatch.setattr(sys, "argv", [relpath, *argv_extra])

    args = script.parse_args()

    assert getattr(args, option) == str(tmp_path / "volume")


@pytest.mark.parametrize(("relpath", "option", "argv_extra"), [
    ("download_datasets.py", "output_dir", []),
    ("upload_datasets.py", "input_dir", ["--token", "t"]),
])
def test_dataset_transfer_default_dir_is_the_old_relative_dir(monkeypatch, relpath, option, argv_extra):
    import sys

    monkeypatch.chdir(CODE_DIR)
    script = load_script(monkeypatch, relpath, ("huggingface_hub",))
    monkeypatch.setattr(sys, "argv", [relpath, *argv_extra])

    args = script.parse_args()

    assert getattr(args, option) == os.path.abspath("./data")
