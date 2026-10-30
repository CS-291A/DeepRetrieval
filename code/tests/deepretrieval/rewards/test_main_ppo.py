"""A3, B14, E33: reward routing, PubMed key loading and the GPU preflight in verl/trainer/main_ppo.py."""
import logging
import os

import pytest
from omegaconf import OmegaConf

from verl.trainer import main_ppo
from verl.utils.apis.pubmed import PubmedAPI


def test_select_rm_score_fn_routes_fiqa_to_fiqa_reward(load_reward_module):
    fiqa = load_reward_module("verl.utils.reward_score.fiqa")

    score_fn = main_ppo._select_rm_score_fn("fiqa_test")

    assert score_fn is fiqa.compute_score


def test_select_rm_score_fn_names_unknown_data_source():
    with pytest.raises(NotImplementedError, match=r"^No reward function for data_source 'trial_no_reason'"):
        main_ppo._select_rm_score_fn("trial_no_reason")


def test_select_rm_score_fn_rejects_removed_wikisql_data_source():
    # wikisql was removed upstream (8c8cbd7); its reward module could only raise ImportError.
    with pytest.raises(NotImplementedError, match=r"^No reward function for data_source 'wikisql_train'"):
        main_ppo._select_rm_score_fn("wikisql_train")


def test_pubmed_key_path_is_resolved_inside_the_package_not_the_cwd():
    expected = os.path.join(main_ppo.project_root, "verl", "utils", "reward_score", "apis", "pubmed_api.key")

    assert main_ppo.PUBMED_API_KEY_PATH == expected


def test_reward_manager_reads_pubmed_key_from_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("secret-key\n")
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(key_file))

    manager = main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert manager.pubmed_api is not None
    assert manager.pubmed_api.api_key == "secret-key"


def test_reward_manager_without_key_file_builds_keyless_pubmed_client(monkeypatch, tmp_path):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))

    manager = main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert isinstance(manager.pubmed_api, PubmedAPI)
    assert manager.pubmed_api.api_key is None


def test_reward_manager_without_key_file_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))

    with caplog.at_level(logging.WARNING, logger=main_ppo.__name__):
        main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert "missing.key" in caplog.text


class _FakeRemoteTask:
    def __init__(self):
        self.configs = []

    def remote(self, config):
        self.configs.append(config)
        return "task-ref"


def _patch_ray(monkeypatch, resources):
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_ppo.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(main_ppo.ray, "cluster_resources", lambda: resources)
    monkeypatch.setattr(main_ppo.ray, "get", lambda ref: ref)
    task = _FakeRemoteTask()
    monkeypatch.setattr(main_ppo, "main_task", task)
    return task


def _config(n_gpus_per_node, nnodes):
    return OmegaConf.create({"trainer": {"n_gpus_per_node": n_gpus_per_node, "nnodes": nnodes}})


def test_main_raises_when_cluster_has_fewer_gpus_than_requested(monkeypatch):
    _patch_ray(monkeypatch, {"CPU": 8.0, "GPU": 4.0})

    with pytest.raises(RuntimeError, match=r"requests 8 GPUs .* has 4"):
        main_ppo.main.__wrapped__(_config(n_gpus_per_node=4, nnodes=2))


def test_main_does_not_start_training_when_gpus_are_missing(monkeypatch):
    task = _patch_ray(monkeypatch, {"CPU": 8.0})

    with pytest.raises(RuntimeError, match=r"requests 1 GPUs"):
        main_ppo.main.__wrapped__(_config(n_gpus_per_node=1, nnodes=1))

    assert task.configs == []


def test_main_starts_training_when_cluster_has_enough_gpus(monkeypatch):
    task = _patch_ray(monkeypatch, {"CPU": 8.0, "GPU": 8.0})
    config = _config(n_gpus_per_node=8, nnodes=1)

    main_ppo.main.__wrapped__(config)

    assert task.configs == [config]
