"""verl modules log through their own named logger, not the root logger (audit LOG002)."""
import logging

import pytest

from tests.deepretrieval.evals._loader import stub_modules


class _Aborted(Exception):
    pass


def test_dead_worker_warning_uses_module_logger(monkeypatch, caplog):
    from verl.single_controller.base import worker_group

    def fake_raise_signal(signum):
        raise _Aborted

    monkeypatch.setattr(worker_group.signal, "raise_signal", fake_raise_signal)

    with caplog.at_level(logging.WARNING), pytest.raises(_Aborted):
        worker_group.check_workers_alive(["w0"], is_alive=lambda w: False, gap_time=0)

    [record] = caplog.records
    assert record.name == "verl.single_controller.base.worker_group"
    assert record.getMessage() == "worker w0 is not alive sending signal to main thread"


@pytest.fixture
def ray_backend(monkeypatch, fresh_import):
    # cupy is CUDA-only; the rendezvous module imports its NCCL bindings at top level.
    stub_modules(monkeypatch, ["cupy", "cupy.cuda", "cupy.cuda.nccl"])
    return fresh_import("verl.utils.rendezvous.ray_backend")


def test_duplicate_actor_warning_uses_module_logger(ray_backend, monkeypatch, caplog):
    actors = [{"name": "g", "namespace": "a"}, {"name": "g", "namespace": "b"}]
    monkeypatch.setattr(ray_backend, "list_named_actors", lambda all_namespaces: actors)

    with caplog.at_level(logging.INFO):
        store = ray_backend.get_nccl_id_store_by_name("g")

    assert store is None
    [record] = caplog.records
    assert record.name == "verl.utils.rendezvous.ray_backend"
    assert record.levelno == logging.WARNING


def test_missing_actor_info_uses_module_logger(ray_backend, monkeypatch, caplog):
    monkeypatch.setattr(ray_backend, "list_named_actors", lambda all_namespaces: [])

    with caplog.at_level(logging.INFO):
        ray_backend.get_nccl_id_store_by_name("g")

    [record] = caplog.records
    assert record.name == "verl.utils.rendezvous.ray_backend"
    assert record.getMessage() == "failed to get any actor named g"


def test_rendezvous_retry_info_uses_module_logger(ray_backend, monkeypatch, caplog):
    monkeypatch.setattr(ray_backend, "get_nccl_id_store_by_name", lambda name: None)
    monkeypatch.setattr(ray_backend.time, "sleep", lambda s: None)

    with caplog.at_level(logging.INFO):
        result = ray_backend.create_nccl_communicator_in_ray(rank=1, world_size=2, group_name="g", max_retries=2)

    assert result is None
    assert [r.name for r in caplog.records] == ["verl.utils.rendezvous.ray_backend"] * 2
