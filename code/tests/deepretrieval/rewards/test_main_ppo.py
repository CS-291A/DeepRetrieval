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


def test_reward_manager_prefers_the_pubmed_key_env_var(monkeypatch, tmp_path):
    key_file = tmp_path / "pubmed_api.key"
    key_file.write_text("file-key\n")
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(key_file))
    monkeypatch.setenv("PUBMED_API_KEY", "env-key")

    manager = main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert manager.pubmed_api is not None
    assert manager.pubmed_api.api_key == "env-key"


def test_reward_manager_reads_the_pubmed_key_env_var_without_a_key_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))
    monkeypatch.setenv("PUBMED_API_KEY", "env-key")

    manager = main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert manager.pubmed_api is not None
    assert manager.pubmed_api.api_key == "env-key"


def test_keyless_warning_names_the_env_var(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))

    with caplog.at_level(logging.WARNING, logger=main_ppo.__name__):
        main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert "PUBMED_API_KEY" in caplog.text


def test_reward_manager_never_logs_the_pubmed_key(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))
    monkeypatch.setenv("PUBMED_API_KEY", "env-key-do-not-log")

    with caplog.at_level(logging.DEBUG):
        main_ppo.RewardManager(tokenizer=None, num_examine=0)

    assert "env-key-do-not-log" not in caplog.text


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


class _FixedDecodeTokenizer:
    def decode(self, ids):
        return "<prompt><answer>q</answer>"


def _nq_sample():
    import numpy as np
    import torch

    from verl import DataProto
    return DataProto.from_dict(
        tensors={"prompts": torch.tensor([[1]]), "responses": torch.tensor([[2]]),
                 "attention_mask": torch.ones(1, 2, dtype=torch.long)},
        non_tensors={"data_source": np.array(["nq_serini"], dtype=object),
                     "reward_model": np.array([{"ground_truth": {"target": ["a"]}}], dtype=object)},
    )


def test_reward_manager_logs_examined_samples_at_info_instead_of_printing(monkeypatch, tmp_path, caplog, capsys):
    monkeypatch.setattr(main_ppo, "PUBMED_API_KEY_PATH", str(tmp_path / "missing.key"))
    manager = main_ppo.RewardManager(tokenizer=_FixedDecodeTokenizer(), num_examine=1)
    monkeypatch.setattr(main_ppo.reward_registry.RewardRoute, "compute_score_fn",
                        lambda self, data_source: lambda **kwargs: 0.0)

    with caplog.at_level(logging.INFO, logger=main_ppo.__name__):
        manager(_nq_sample())

    info_records = [r.getMessage() for r in caplog.records
                    if r.name == main_ppo.__name__ and r.levelno == logging.INFO]
    assert info_records == ["<prompt><answer>q</answer>"]
    assert capsys.readouterr().out == ""


class _StopAfterLoggingSetup(Exception):
    pass


def test_main_configures_logging_before_starting_ray(monkeypatch):
    events = []
    _patch_ray(monkeypatch, {"GPU": 1.0})
    monkeypatch.setattr(main_ppo.ray, "init", lambda **kwargs: events.append("ray.init"))
    monkeypatch.setattr(main_ppo, "configure_logging", lambda: events.append("configure_logging"))

    main_ppo.main.__wrapped__(_config(n_gpus_per_node=1, nnodes=1))

    assert events == ["configure_logging", "ray.init"]


def test_main_checks_the_triton_toolchain_before_starting_ray(monkeypatch):
    events = []
    _patch_ray(monkeypatch, {"GPU": 1.0})
    monkeypatch.setattr(main_ppo.ray, "init", lambda **kwargs: events.append("ray.init"))
    monkeypatch.setattr(main_ppo, "require_triton_toolchain", lambda: events.append("require_triton_toolchain"))

    main_ppo.main.__wrapped__(_config(n_gpus_per_node=1, nnodes=1))

    assert events == ["require_triton_toolchain", "ray.init"]


def test_main_task_configures_logging_in_the_ray_worker_first(monkeypatch):
    # Ray workers do not inherit the driver's logging config, and rewards are computed in main_task.
    def configure_logging():
        raise _StopAfterLoggingSetup

    monkeypatch.setattr(main_ppo, "configure_logging", configure_logging)

    with pytest.raises(_StopAfterLoggingSetup):
        # ray.remote keeps the undecorated function on the RemoteFunction (not in Ray's type stubs)
        vars(main_ppo.main_task)["_function"](OmegaConf.create({}))
