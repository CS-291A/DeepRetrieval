"""src/eval PubMed scripts and data_preprocess/utils/pubdate.py read $PUBMED_API_KEY first.

The key file (verl/utils/reward_score/apis/pubmed_api.key, resolved from the package) stays as
fallback; the eval scripts still refuse to run keyless, naming both sources.
"""
import sys

import pytest

from tests.deepretrieval.evals._loader import (
    load_script,
    run_script_as_main,
    stub_modules,
)
from verl.utils import credentials

LLM_CLIENT_STUBS = ("src.utils.claude_api", "src.utils.gpt_azure")
PUBMED_MAIN_SCRIPTS = (
    ("src/eval/SearchEngine/eval_pubmed.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_no_reason.py", ()),
    ("src/eval/SearchEngine/eval_ctgov.py", ("--api", "pubmed")),
    ("src/eval/SearchEngine/eval_pubmed_base.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_reason.py", ()),
    ("src/eval/SearchEngine/eval_pubmed_no_reason_gpt.py", ()),
)


class _Constructed(Exception):
    """Raised by the PubmedAPI stand-in so main() stops right after choosing the key."""


@pytest.fixture
def no_key_file(monkeypatch, tmp_path):
    monkeypatch.setattr(credentials, "DEFAULT_PUBMED_API_KEY_PATH", tmp_path / "missing.key")


def _load_with_recording_client(monkeypatch, relpath, argv):
    script = load_script(monkeypatch, relpath, LLM_CLIENT_STUBS)
    monkeypatch.setattr(sys, "argv", [relpath, *argv])
    monkeypatch.setattr(script, "load_model", lambda path: (object(), object()), raising=False)
    keys = []

    def recording_client(*args, **kwargs):
        keys.append(kwargs.get("api_key"))
        raise _Constructed

    monkeypatch.setattr(script, "PubmedAPI", recording_client)
    return script, keys


@pytest.mark.parametrize(("relpath", "argv"), PUBMED_MAIN_SCRIPTS)
def test_pubmed_eval_uses_the_env_var_key(monkeypatch, no_key_file, relpath, argv):
    monkeypatch.setenv("PUBMED_API_KEY", "env-key")
    script, keys = _load_with_recording_client(monkeypatch, relpath, argv)

    with pytest.raises(_Constructed):
        script.main()

    assert keys == ["env-key"]


@pytest.mark.parametrize(("relpath", "argv"), PUBMED_MAIN_SCRIPTS)
def test_pubmed_eval_falls_back_to_the_package_key_file_from_any_cwd(monkeypatch, tmp_path, relpath, argv):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("file-key\n")
    monkeypatch.setattr(credentials, "DEFAULT_PUBMED_API_KEY_PATH", key_file)
    monkeypatch.chdir(tmp_path / "..")
    script, keys = _load_with_recording_client(monkeypatch, relpath, argv)

    with pytest.raises(_Constructed):
        script.main()

    assert keys == ["file-key"]


@pytest.mark.parametrize(("relpath", "argv"), PUBMED_MAIN_SCRIPTS)
def test_pubmed_eval_without_any_key_names_env_var_and_key_file(monkeypatch, no_key_file, relpath, argv):
    script, _ = _load_with_recording_client(monkeypatch, relpath, argv)

    with pytest.raises(FileNotFoundError, match="PUBMED_API_KEY") as excinfo:
        script.main()

    assert "missing.key" in str(excinfo.value)


PUBDATE = "data_preprocess/utils/pubdate.py"
API_STUBS = ("verl.utils.apis.pubmed", "verl.utils.apis.ctgov")


def test_pubdate_uses_the_env_var_key(monkeypatch, no_key_file):
    monkeypatch.setenv("PUBMED_API_KEY", "env-key")
    stubs = stub_modules(monkeypatch, API_STUBS)
    stubs["verl.utils.apis.pubmed"].PubmedAPI.side_effect = _Constructed

    with pytest.raises(_Constructed):
        run_script_as_main(monkeypatch, PUBDATE)

    stubs["verl.utils.apis.pubmed"].PubmedAPI.assert_called_once_with(api_key="env-key")
