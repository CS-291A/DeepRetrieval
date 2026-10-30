# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""
import itertools

import torch
import torch.distributed
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.import_utils import flash_attn_import_error
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean, scalars_to_floats
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

try:
    from flash_attn.bert_padding import (  # pyright: ignore[reportMissingImports]  # flash-attn is Linux/CUDA-only (gpu extra)
        index_first_axis,
        pad_input,
        rearrange,
        unpad_input,
    )
except ImportError as e:
    raise flash_attn_import_error('DataParallelPPOCritic') from e

__all__ = ['DataParallelPPOCritic']


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')
        print(f'Critic ppo_mini_batch_size={self.config.ppo_mini_batch_size}')
        print(f'Critic ppo_micro_batch_size={self.config.ppo_micro_batch_size}')
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                pad_size = 0  # only read when ulysses_sequence_parallel_size > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1:-1]
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
                values = values[:, -response_length - 1:-1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        assert batch is not None, 'select with batch_keys keeps the tensor batch'
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        indices: list[list[int]] = []  # only read when use_dynamic_bsz
        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        assert data.batch is not None, 'compute_values input carries the rollout tensors'
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length - 1:-1]

        if use_dynamic_bsz:
            flat_indices = list(itertools.chain.from_iterable(indices))
            assert len(flat_indices) == values.size(0), f"{len(flat_indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)
            values = values[revert_indices]

        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        batch = data.select(batch_keys=select_keys).batch
        assert batch is not None, 'select with batch_keys keeps the tensor batch'
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for batch_idx, mini_batch in enumerate(dataloader):
            # split batch into micro_batches
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.critic_optimizer.zero_grad()

            for micro_batch in micro_batches:
                micro_batch = micro_batch.cuda()  # critic device is cpu when using offload
                input_ids = micro_batch['input_ids']
                responses = micro_batch['responses']
                attention_mask = micro_batch['attention_mask']
                position_ids = micro_batch['position_ids']
                values = micro_batch['values']
                returns = micro_batch['returns']
                response_length = responses.size(1)

                eos_mask = attention_mask[:, -response_length - 1:-1]

                vpreds = self._forward_micro_batch(micro_batch)

                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()

                # one device-to-host sync for all per-micro-batch scalars
                vf_loss_value, vf_clipfrac_value, vpred_mean = scalars_to_floats(
                    [vf_loss, vf_clipfrac, masked_mean(vpreds, eos_mask)])
                micro_metrics = {
                    'critic/vf_loss': vf_loss_value,
                    'critic/vf_clipfrac': vf_clipfrac_value,
                    'critic/vpred_mean': vpred_mean,
                }

                append_to_dict(metrics, micro_metrics)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {'critic/grad_norm': grad_norm.detach().item()})
        self.critic_optimizer.zero_grad()
        return metrics
