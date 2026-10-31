"""verl.utils.credentials: the PubMed key comes from $PUBMED_API_KEY, else the key file, else none."""
import logging

import pytest

from verl.utils import credentials

SECRET = "s3cr3t-ncbi-key"


def test_env_var_key_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBMED_API_KEY", SECRET)

    assert credentials.load_pubmed_api_key(tmp_path / "missing.key") == SECRET


def test_env_var_takes_precedence_over_the_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("file-key\n")
    monkeypatch.setenv("PUBMED_API_KEY", SECRET)

    assert credentials.load_pubmed_api_key(key_file) == SECRET


def test_key_file_is_the_fallback(tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("  file-key\n")

    assert credentials.load_pubmed_api_key(key_file) == "file-key"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_env_var_falls_back_to_the_key_file(monkeypatch, tmp_path, blank):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("file-key")
    monkeypatch.setenv("PUBMED_API_KEY", blank)

    assert credentials.load_pubmed_api_key(key_file) == "file-key"


def test_no_env_var_and_no_file_means_keyless(tmp_path):
    assert credentials.load_pubmed_api_key(tmp_path / "missing.key") is None


def test_empty_key_file_means_keyless(tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("\n")

    assert credentials.load_pubmed_api_key(key_file) is None


def test_default_key_path_is_inside_the_package_not_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = credentials.Path(credentials.__file__).resolve().parent / "reward_score" / "apis" / "pubmed_api.key"

    assert credentials.DEFAULT_PUBMED_API_KEY_PATH == expected
    assert expected.is_absolute()


def test_default_key_path_is_read_when_no_path_is_given(monkeypatch, tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("file-key")
    monkeypatch.setattr(credentials, "DEFAULT_PUBMED_API_KEY_PATH", key_file)

    assert credentials.load_pubmed_api_key() == "file-key"


def test_env_key_source_is_logged_but_the_key_never_is(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PUBMED_API_KEY", SECRET)

    with caplog.at_level(logging.DEBUG, logger=credentials.__name__):
        credentials.load_pubmed_api_key(tmp_path / "missing.key")

    assert SECRET not in caplog.text
    assert "PUBMED_API_KEY" in caplog.text


def test_file_key_source_is_logged_but_the_key_never_is(tmp_path, caplog):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text(SECRET)

    with caplog.at_level(logging.DEBUG, logger=credentials.__name__):
        credentials.load_pubmed_api_key(key_file)

    assert SECRET not in caplog.text
    assert str(key_file) in caplog.text
