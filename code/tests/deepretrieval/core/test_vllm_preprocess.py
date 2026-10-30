"""vLLM rollout prompt pre-processing (audit item D26): one host copy per batch, identical output."""
import sys
import types

import pytest
import torch

PAD = 0


@pytest.fixture
def vllm_rollout(monkeypatch, fresh_import):
    # vllm is CUDA-only; the pre-processing helpers never touch it, so stub the imports.
    third_party_vllm = types.ModuleType("verl.third_party.vllm")
    vars(third_party_vllm).update(LLM=object, vllm_version="stub", parallel_state=types.ModuleType("parallel_state"))
    vllm = types.ModuleType("vllm")
    vars(vllm)["SamplingParams"] = object
    monkeypatch.setitem(sys.modules, "verl.third_party.vllm", third_party_vllm)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    return fresh_import("verl.workers.rollout.vllm_rollout.vllm_rollout", "verl.workers.rollout.vllm_rollout")


@pytest.fixture
def left_padded_prompts():
    prompts = torch.randint(1, 100, (6, 8))
    prompts[0, :3] = PAD  # left padding
    prompts[1, :7] = PAD  # a single real token
    prompts[2, 4] = PAD  # a pad id inside the prompt must be kept
    prompts[3, :2] = PAD
    prompts[3, 5] = PAD
    return prompts


def test_batch_preprocess_equals_per_row_preprocess(vllm_rollout, left_padded_prompts):
    expected = [vllm_rollout._pre_process_inputs(PAD, row) for row in left_padded_prompts]

    result = vllm_rollout._pre_process_inputs_batch(PAD, left_padded_prompts)

    assert result == expected
    assert result[1] == [left_padded_prompts[1, 7].item()]


def test_batch_preprocess_rejects_all_pad_prompt(vllm_rollout, left_padded_prompts):
    left_padded_prompts[4] = PAD

    with pytest.raises(ValueError, match="non-pad token"):
        vllm_rollout._pre_process_inputs_batch(PAD, left_padded_prompts)
