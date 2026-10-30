"""download_datasets.py: hf_hub_download is called only with arguments it still honours."""
import pytest

from tests.deepretrieval.evals._loader import load_script


@pytest.fixture
def downloader(monkeypatch):
    return load_script(monkeypatch, "download_datasets.py")


def test_download_omits_deprecated_symlink_argument(downloader, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(downloader, "hf_hub_download", lambda **kwargs: calls.append(kwargs))

    downloader.download_files("org/repo", ["sql/bird/test.parquet"], str(tmp_path))

    assert calls == [{
        "repo_id": "org/repo",
        "filename": "sql/bird/test.parquet",
        "repo_type": "dataset",
        "local_dir": str(tmp_path),
        "token": None,
    }]
