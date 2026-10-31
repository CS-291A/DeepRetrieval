"""Failures in the dataset download/upload scripts are logged with their traceback (audit LOG006)."""
import logging

import pytest

from tests.deepretrieval.evals._loader import load_script


def _boom(*args, **kwargs):
    raise RuntimeError("hub unreachable")


def _error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.fixture
def downloader(monkeypatch):
    return load_script(monkeypatch, "download_datasets.py")


@pytest.fixture
def uploader(monkeypatch):
    return load_script(monkeypatch, "upload_datasets.py")


def test_list_files_failure_logs_traceback(downloader, monkeypatch, caplog):
    monkeypatch.setattr(downloader, "list_repo_files", _boom)

    with caplog.at_level(logging.INFO):
        files = downloader.list_available_files("org/repo")

    assert files == []
    [record] = _error_records(caplog)
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_single_file_download_failure_logs_traceback(downloader, monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(downloader, "hf_hub_download", _boom)

    with caplog.at_level(logging.INFO):
        result = downloader.download_files("org/repo", ["sql/bird/test.parquet"], str(tmp_path))

    assert result == (0, 1)
    [record] = _error_records(caplog)
    assert record.exc_info is not None


def test_category_download_failure_logs_traceback(downloader, monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(downloader, "list_available_files", _boom)

    with caplog.at_level(logging.INFO):
        downloaded = downloader.download_category("org/repo", "sql", str(tmp_path))

    assert downloaded == 0
    [record] = _error_records(caplog)
    assert record.exc_info is not None


def test_repo_creation_failure_logs_traceback(uploader, monkeypatch, caplog):
    monkeypatch.setattr(uploader, "create_repo", _boom)

    with caplog.at_level(logging.INFO):
        created = uploader.create_huggingface_repo("org/repo", token="t")

    assert created is False
    [record] = _error_records(caplog)
    assert record.exc_info is not None


def test_upload_failure_logs_traceback(uploader, monkeypatch, caplog, tmp_path):
    (tmp_path / "a.parquet").write_text("x")
    monkeypatch.setattr(uploader, "upload_folder", _boom)

    with caplog.at_level(logging.INFO):
        uploaded = uploader.upload_category(str(tmp_path), "sql", "org/repo", token="t")

    assert uploaded is False
    [record] = _error_records(caplog)
    assert record.exc_info is not None
