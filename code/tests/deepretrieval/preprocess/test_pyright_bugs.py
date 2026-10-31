"""Runtime bugs in data_preprocess/ that pyright reported as possibly-unbound variables."""
import builtins
import json
from pathlib import Path

import pytest

from tests.deepretrieval.evals._loader import (
    load_script,
    run_script_as_main,
    stub_modules,
)
from verl.utils import credentials

PUBDATE = "data_preprocess/utils/pubdate.py"
API_STUBS = ("verl.utils.apis.pubmed", "verl.utils.apis.ctgov")


def test_get_pub_date_reraises_original_error_when_failing_before_first_item(monkeypatch, tmp_path):
    """Zero items to process + a failing final save used to raise UnboundLocalError on `i`."""
    pubdate = load_script(monkeypatch, PUBDATE, stubs=API_STUBS)
    output_file = tmp_path / "missing_dir" / "out.jsonl"  # parent absent: the save raises

    with pytest.raises(FileNotFoundError):
        pubdate.get_pub_date([], str(output_file), str(tmp_path / "ckpt.json"), pubmed_api=None)

    assert not output_file.exists()


def test_pubdate_main_runs_without_api_key_file(monkeypatch, tmp_path, capsys):
    """Without verl/.../pubmed_api.key, `pubmed_api` was never bound and every split failed."""
    monkeypatch.chdir(tmp_path)  # no key file under the cwd
    monkeypatch.setattr(credentials, "DEFAULT_PUBMED_API_KEY_PATH", tmp_path / "missing.key")  # nor in the package
    real_open = builtins.open
    remote_prefix = "/home/pj20/server-04/LMR/code/data/raw_data/ctgov/"
    for split in ("train", "test", "dev"):  # audit: ignore[BRANCHING] arrange: one fixture file per split
        row = {"origin": {"pmid": "1"}, "pub_date": "2020/01/01"}  # already dated: no API call
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")

    def redirecting_open(file, *args, **kwargs):
        name = str(file)
        if name.startswith(remote_prefix):
            file = tmp_path / Path(name).name
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", redirecting_open)
    stubs = stub_modules(monkeypatch, API_STUBS)

    run_script_as_main(monkeypatch, PUBDATE)

    out = capsys.readouterr().out
    assert "Error processing" not in out
    for split in ("train", "test", "dev"):
        assert f"Successfully processed {split} split" in out
    stubs["verl.utils.apis.pubmed"].PubmedAPI.assert_called_once_with(api_key=None)
