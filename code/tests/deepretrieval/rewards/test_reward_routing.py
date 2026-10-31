"""Characterisation of reward routing by data_source (audit item 15).

Pins, for every data_source the training code can route (names from data_preprocess/ and
scripts/train/), which `compute_score` RewardManager calls and with which keyword arguments.
Written against the three hand-maintained substring chains before they were replaced by
verl/trainer/reward_registry.py; the refactor had to keep every case green.

Each reward module is replaced by a recording fake in `sys.modules` and on its parent
package, so both `from pkg import mod` and `importlib.import_module` resolve to the fake.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import verl.utils.reward_score
import verl.utils.reward_score_dense
from verl import DataProto
from verl.trainer import main_ppo

SPARSE = "verl.utils.reward_score"
DENSE = "verl.utils.reward_score_dense"

SPARSE_MODULES = ("screening", "pubmed", "pmd_no_reason", "ctgov", "fiqa", "scifact", "nfcorpus", "nq_serini",
                  "nq_no_reason", "triviaqa", "tqa_no_reason", "squad", "squad_no_reason", "hotpotqa", "fever",
                  "msmarco_beir", "msmarco", "bird", "bird_no_reason", "spider", "spider_no_reason")
DENSE_MODULES = ("scifact", "nfcorpus", "hotpotqa", "fever", "msmarco_beir")

GROUND_TRUTH = {"target": ["doc-1"]}
PUB_DATE = "2020/01/01"
DB_PATH = str(Path(__file__).resolve())  # any existing absolute file: sql_kwargs rejects a missing database
DECODED = "<decoded prompt and response>"

# data_source -> (module that must score it, kind of extra kwargs it must receive)
ROUTES = {
    "screening": (f"{SPARSE}.screening", "none"),
    "pubmed": (f"{SPARSE}.pubmed", "publication"),
    "pubmed_32": (f"{SPARSE}.pubmed", "publication"),
    "pmd_no_reason": (f"{SPARSE}.pmd_no_reason", "publication"),
    "ctgov": (f"{SPARSE}.ctgov", "trial"),
    "ctgov_32": (f"{SPARSE}.ctgov", "trial"),
    "fiqa_train": (f"{SPARSE}.fiqa", "data_source"),
    "fiqa_test": (f"{SPARSE}.fiqa", "data_source"),
    "scifact_test": (f"{SPARSE}.scifact", "data_source"),
    "scifact_dense_test": (f"{DENSE}.scifact", "data_source"),
    "nfcorpus_test": (f"{SPARSE}.nfcorpus", "data_source"),
    "nfcorpus_dense_test": (f"{DENSE}.nfcorpus", "data_source"),
    "hotpotqa_test": (f"{SPARSE}.hotpotqa", "data_source"),
    "hotpotqa_dense_test": (f"{DENSE}.hotpotqa", "data_source"),
    "fever_test": (f"{SPARSE}.fever", "data_source"),
    "fever_dense_test": (f"{DENSE}.fever", "data_source"),
    "msmarco_beir_test": (f"{SPARSE}.msmarco_beir", "data_source"),
    "msmarco_beir_dense_test": (f"{DENSE}.msmarco_beir", "data_source"),
    # msmarco subsets have no dense module: the sparse module reads the mode from data_source
    "msmarco_health_sparse_test": (f"{SPARSE}.msmarco", "data_source"),
    "msmarco_tech_dense_test": (f"{SPARSE}.msmarco", "data_source"),
    "nq_serini": (f"{SPARSE}.nq_serini", "none"),
    "nq_serini_no_reason": (f"{SPARSE}.nq_no_reason", "none"),
    "triviaqa": (f"{SPARSE}.triviaqa", "none"),
    "triviaqa_no_reason": (f"{SPARSE}.tqa_no_reason", "none"),
    "squad": (f"{SPARSE}.squad", "none"),
    "squad_no_reason": (f"{SPARSE}.squad_no_reason", "none"),
    "bird_train": (f"{SPARSE}.bird", "sql"),
    "bird_wor_train": (f"{SPARSE}.bird_no_reason", "sql"),
    "spider_train": (f"{SPARSE}.spider", "sql"),
    "spider_wor_train": (f"{SPARSE}.spider_no_reason", "sql"),
}

UNROUTABLE = ("trial_no_reason", "wikisql_train", "sql_all_train", "unknown", "")


@pytest.fixture
def score_calls(monkeypatch):
    """Replace every reward module with a fake whose compute_score records (module name, kwargs)."""
    calls = []
    for package, names in ((verl.utils.reward_score, SPARSE_MODULES), (verl.utils.reward_score_dense, DENSE_MODULES)):
        for name in names:
            full_name = f"{package.__name__}.{name}"
            fake = types.ModuleType(full_name)

            def compute_score(_module=full_name, **kwargs):
                calls.append((_module, kwargs))
                return 1.0

            vars(fake)["compute_score"] = compute_score
            monkeypatch.setitem(sys.modules, full_name, fake)
            monkeypatch.setattr(package, name, fake, raising=False)
    return calls


class _DecodeTokenizer:
    def decode(self, ids):
        return DECODED


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))
    return main_ppo.RewardManager(tokenizer=_DecodeTokenizer(), num_examine=0)


def _one_sample(data_source):
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[5, 6]]),
            "responses": torch.tensor([[7, 8, 9]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array([data_source], dtype=object),
            "reward_model": np.array([{"ground_truth": GROUND_TRUTH}], dtype=object),
            "pub_date": np.array([PUB_DATE], dtype=object),
            "extra_info": np.array([{"db_path": DB_PATH}], dtype=object),
        },
    )


def _expected_kwargs(kind, data_source, manager):
    base = {"solution_str": DECODED, "ground_truth": GROUND_TRUTH}
    extra = {
        "none": {},
        "publication": {"search_api": manager.pubmed_api, "literature_type": "publication", "pub_date": PUB_DATE},
        "trial": {"search_api": manager.ctgov_api, "literature_type": "trial", "pub_date": PUB_DATE},
        "data_source": {"data_source": data_source},
        "sql": {"data_source": data_source, "db_path": DB_PATH},
    }[kind]
    return base | extra


@pytest.mark.parametrize(("data_source", "module"), [(ds, module) for ds, (module, _) in ROUTES.items()])
def test_select_rm_score_fn_picks_the_reward_module(score_calls, data_source, module):
    score_fn = main_ppo._select_rm_score_fn(data_source)

    assert score_fn is sys.modules[module].compute_score


@pytest.mark.parametrize(("data_source", "module", "kind"), [(ds, m, k) for ds, (m, k) in ROUTES.items()])
def test_reward_manager_passes_the_routed_kwargs(score_calls, manager, data_source, module, kind):
    manager(_one_sample(data_source))

    assert score_calls == [(module, _expected_kwargs(kind, data_source, manager))]


def test_reward_manager_places_the_score_on_the_last_valid_response_token(score_calls, manager):
    reward = manager(_one_sample("nq_serini"))

    assert reward.tolist() == [[0.0, 0.0, 1.0]]


@pytest.mark.parametrize("data_source", UNROUTABLE)
def test_select_rm_score_fn_rejects_unroutable_data_source(score_calls, data_source):
    with pytest.raises(NotImplementedError, match=rf"^No reward function for data_source '{data_source}'$"):
        main_ppo._select_rm_score_fn(data_source)
