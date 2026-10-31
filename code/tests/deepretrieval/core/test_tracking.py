"""W&B runs are finished with the training outcome (audit item 5).

`fit` runs inside the Ray `main_task`, where Ray catches the exception before
sys.excepthook, so wandb's own crash hook never fires: the trainer must call
`Tracking.finish(exit_code=...)` itself. A fake `wandb` module stands in (no network).
"""
import sys
import types

import pytest

from tests.deepretrieval.core._fit_harness import (
    fit_config,
    make_fit_trainer,
    run_fit,
    stub_step_metrics,
)
from verl.utils.tracking import Tracking


@pytest.fixture
def wandb_events(monkeypatch):
    """Install a fake `wandb` module; return the list of calls it receives."""
    events = []
    wandb = types.ModuleType("wandb")

    def init(**kwargs):
        events.append(("init", kwargs["project"]))

    def log(data, step):
        events.append(("log", step))

    def finish(exit_code=None):
        events.append(("finish", exit_code))

    vars(wandb).update(init=init, log=log, finish=finish, login=lambda key: None)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return events


def _finishes(events):
    return [event for event in events if event[0] == "finish"]


def test_finish_passes_the_exit_code_to_wandb(wandb_events):
    tracker = Tracking(project_name="p", experiment_name="e", default_backend="wandb")

    tracker.finish(exit_code=1)

    assert _finishes(wandb_events) == [("finish", 1)]


def test_finish_without_wandb_backend_is_a_no_op(wandb_events):
    tracker = Tracking(project_name="p", experiment_name="e", default_backend="console")

    tracker.finish(exit_code=0)

    assert wandb_events == []


class _TrainingCrashed(RuntimeError):
    pass


def _crashing_reward_fn(batch):
    raise _TrainingCrashed("reward server went away")


def test_fit_marks_the_run_failed_and_reraises_when_training_crashes(monkeypatch, wandb_events):
    stub_step_metrics(monkeypatch)
    trainer = make_fit_trainer(fit_config("data/sql/bird/train.parquet", logger="wandb"), [1.0])
    trainer.reward_fn = _crashing_reward_fn

    with pytest.raises(_TrainingCrashed, match="^reward server went away$"):
        run_fit(trainer)

    assert wandb_events[-1] == ("finish", 1)
    assert _finishes(wandb_events) == [("finish", 1)]


def test_fit_finishes_the_run_with_exit_code_zero_after_the_last_step(monkeypatch, wandb_events):
    stub_step_metrics(monkeypatch)

    run_fit(make_fit_trainer(fit_config("data/sql/bird/train.parquet", logger="wandb"), [1.0]))

    assert wandb_events == [("init", "p"), ("log", 1), ("finish", 0)]


def test_fit_finishes_the_run_with_exit_code_zero_after_validation_only(monkeypatch, wandb_events):
    config = fit_config("data/sql/bird/train.parquet", logger="wandb", val_only=True)
    trainer = make_fit_trainer(config, [1.0], val_reward_fn=lambda batch: None)
    trainer._validate = lambda: {"val/test_score/bird": 0.5}

    run_fit(trainer)

    assert wandb_events == [("init", "p"), ("log", 0), ("finish", 0)]
