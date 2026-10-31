"""verl.utils.paths: where the datasets and indexes live (DEEPRETRIEVAL_DATA_DIR or <code>/data)."""
from pathlib import Path

import pytest

from verl.utils import paths

CODE_DIR = Path(__file__).resolve().parents[3]


def test_default_data_dir_is_the_code_dirs_data_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the default must not follow the cwd

    assert paths.data_dir() == CODE_DIR / "data"


def test_default_data_dir_matches_the_old_cwd_relative_data_folder(monkeypatch):
    monkeypatch.chdir(CODE_DIR)  # shell scripts used to rely on cwd = code/

    assert paths.data_dir() == Path("data").resolve()


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path / "volume"))

    assert paths.data_dir() == tmp_path / "volume"


def test_env_var_with_tilde_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", "~/dr-data")

    assert paths.data_dir() == tmp_path / "dr-data"


def test_relative_env_var_is_made_absolute_against_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", "rel/data")

    result = paths.data_dir()

    assert result.is_absolute()
    assert result == tmp_path.resolve() / "rel" / "data"


def test_empty_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", "")

    assert paths.data_dir() == CODE_DIR / "data"


def test_data_path_joins_parts_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path))

    assert paths.data_path("search_engine", "pubmed_32/train.parquet") == tmp_path / "search_engine" / "pubmed_32" / "train.parquet"


def test_data_path_rejects_absolute_parts():
    with pytest.raises(ValueError, match="relative"):
        paths.data_path("/etc/passwd")


def test_data_path_rejects_escaping_the_data_dir():
    with pytest.raises(ValueError, match="outside"):
        paths.data_path("..", "secrets")


@pytest.mark.parametrize("stored", ["data/sql/bird/x.sqlite", "./data/sql/bird/x.sqlite"])
def test_resolve_data_reference_maps_stored_data_prefix_under_data_dir(monkeypatch, tmp_path, stored):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path))

    assert paths.resolve_data_reference(stored) == tmp_path / "sql" / "bird" / "x.sqlite"


def test_resolve_data_reference_keeps_absolute_paths(tmp_path):
    assert paths.resolve_data_reference(str(tmp_path / "db.sqlite")) == tmp_path / "db.sqlite"


def test_resolve_data_reference_keeps_other_relative_paths():
    assert paths.resolve_data_reference("demo.sqlite") == Path("demo.sqlite")


def test_require_existing_returns_the_path_when_present(tmp_path):
    target = tmp_path / "present.txt"
    target.write_text("x")

    assert paths.require_existing(target, "BM25 index") == target


def test_require_existing_names_the_path_and_the_env_var(tmp_path):
    missing = tmp_path / "absent"

    with pytest.raises(FileNotFoundError, match="DEEPRETRIEVAL_DATA_DIR") as excinfo:
        paths.require_existing(missing, "BM25 index")

    assert str(missing) in str(excinfo.value)
    assert "BM25 index" in str(excinfo.value)


def test_stored_data_reference_round_trips_through_resolve(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPRETRIEVAL_DATA_DIR", str(tmp_path))

    stored = paths.stored_data_reference("raw_data/bird/dev/db.sqlite")

    assert not Path(stored).is_absolute()  # parquet files stay portable between machines
    assert paths.resolve_data_reference(stored) == tmp_path / "raw_data" / "bird" / "dev" / "db.sqlite"


def test_stored_data_reference_matches_the_format_old_parquets_used():
    assert paths.stored_data_reference("raw_data/spider/x.sqlite") == "data/raw_data/spider/x.sqlite"
