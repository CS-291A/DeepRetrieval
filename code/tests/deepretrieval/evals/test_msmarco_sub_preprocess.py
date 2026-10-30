"""data_preprocess/{,no_reason/BM25/}msmarco_sub.py: one dataset load per domain, both modes written."""
import builtins
import json

import pandas as pd
import pytest

from tests.deepretrieval.evals._loader import run_script_as_main

DOMAINS = ("all", "health", "science", "tech")
SCRIPTS = (
    ("data_preprocess/msmarco_sub.py", "data/local_index_search"),
    ("data_preprocess/no_reason/BM25/msmarco_sub.py", "data/local_index_search/no_reason"),
)


@pytest.fixture
def raw_msmarco(tmp_path, monkeypatch):
    """Tiny data/raw_data/msmarco/msmarco_<domain>/{train,dev}.jsonl under a fresh cwd."""
    monkeypatch.chdir(tmp_path)
    for domain in DOMAINS:
        folder = tmp_path / "data" / "raw_data" / "msmarco" / f"msmarco_{domain}"
        folder.mkdir(parents=True)
        for split, n in (("train", 3), ("dev", 2)):
            lines = [json.dumps({"question": f"{domain} q{i}", "docs_id": [str(i)]}) for i in range(n)]
            (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


@pytest.fixture
def train_file_opens(monkeypatch):
    """Count opens of any */train.jsonl."""
    opened = []
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        if str(file).endswith("train.jsonl"):
            opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    return opened


@pytest.mark.parametrize(("relpath", "out_dir"), SCRIPTS)
def test_each_domain_is_loaded_once(raw_msmarco, train_file_opens, monkeypatch, relpath, out_dir):
    run_script_as_main(monkeypatch, relpath)

    assert sorted(train_file_opens) == sorted(
        f"data/raw_data/msmarco/msmarco_{d}/train.jsonl" for d in DOMAINS
    )


@pytest.mark.parametrize(("relpath", "out_dir"), SCRIPTS)
@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_each_mode_output_is_tagged_and_prompted_for_that_mode(raw_msmarco, monkeypatch, relpath,
                                                               out_dir, mode):
    run_script_as_main(monkeypatch, relpath)

    frame = pd.read_parquet(raw_msmarco / out_dir / "msmarco_health" / mode / "train.parquet")
    prompt = frame["prompt"].iloc[0][0]["content"]
    assert set(frame["data_source"]) == {f"msmarco_{mode}_train"}
    assert f"corpus with {mode} retrieval" in prompt
