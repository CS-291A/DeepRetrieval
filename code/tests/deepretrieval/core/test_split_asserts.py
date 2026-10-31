"""Compound asserts split so the failure names the half that broke (audit NSP004).

Same conditions and exception type as before; only the message changed.
"""
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.optim.sgd import SGD

from verl.protocol import DataProto
from verl.single_controller.base.decorator import dispatch_dp_compute
from verl.single_controller.base.worker_group import WorkerGroup
from verl.utils.model import create_random_mask
from verl.utils.torch_functional import get_cosine_schedule_with_warmup


class _TwoWorkerGroup(WorkerGroup):
    def __init__(self):  # no resource pool or workers needed to dispatch arguments
        self._workers = [object(), object()]


def test_non_tensor_batch_wrong_dtype_names_the_key_and_dtype():
    batch = TensorDict({"x": torch.zeros(2, 1)}, batch_size=[2])

    with pytest.raises(AssertionError, match=r"non_tensor_batch\['tag'\] must have dtype=object, got int64"):
        DataProto(batch=batch, non_tensor_batch={"tag": np.array([1, 2], dtype=np.int64)})


def test_dp_dispatch_non_sequence_arg_names_the_type():
    with pytest.raises(AssertionError, match="must be a tuple or list, got int"):
        dispatch_dp_compute(_TwoWorkerGroup(), 5)


def test_dp_dispatch_wrong_length_arg_names_the_world_size():
    with pytest.raises(AssertionError, match="length 3 != world_size 2"):
        dispatch_dp_compute(_TwoWorkerGroup(), [1, 2, 3])


def test_dp_dispatch_wrong_length_kwarg_names_the_key():
    with pytest.raises(AssertionError, match="kwarg 'data' has length 1 != world_size 2"):
        dispatch_dp_compute(_TwoWorkerGroup(), data=[1])


def test_dp_dispatch_well_formed_args_pass_through():
    group = _TwoWorkerGroup()

    assert dispatch_dp_compute(group, [1, 2], data=(3, 4)) == (([1, 2],), {"data": (3, 4)})


@pytest.mark.parametrize(
    ("valid", "match"),
    [(0.0, r"max_ratio_of_valid_token must be > 0"), (1.5, r"max_ratio_of_valid_token must be <= 1")],
)
def test_random_mask_valid_ratio_names_the_broken_bound(valid, match):
    with pytest.raises(AssertionError, match=match):
        create_random_mask(torch.zeros(2, 8, dtype=torch.long), max_ratio_of_valid_token=valid, max_ratio_of_left_padding=0.0)


@pytest.mark.parametrize(
    ("padding", "match"),
    [(-0.1, r"max_ratio_of_left_padding must be >= 0"), (1.0, r"max_ratio_of_left_padding must be < 1")],
)
def test_random_mask_padding_ratio_names_the_broken_bound(padding, match):
    with pytest.raises(AssertionError, match=match):
        create_random_mask(torch.zeros(2, 8, dtype=torch.long), max_ratio_of_valid_token=0.5, max_ratio_of_left_padding=padding)


@pytest.mark.parametrize(("ratio", "match"), [(-0.1, r"min_lr_ratio must be >= 0"), (1.1, r"min_lr_ratio must be <= 1")])
def test_cosine_schedule_min_lr_ratio_names_the_broken_bound(ratio, match):
    optimizer = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)

    with pytest.raises(AssertionError, match=match):
        get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=10, min_lr_ratio=ratio)
