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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import logging
import os
import sys
from collections.abc import Callable

# RAY_ENABLE_UV_RUN_RUNTIME_ENV defaults to "0" in verl/__init__.py: the package is imported (and
# imports ray) before this module body runs, so setting it here would be too late.

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  
sys.path.insert(0, project_root)  

import utils.java_init  # noqa: F401  (imported for its side effect: sets JAVA_HOME)

# utils.java_init must run before pyserini is imported; keep import sorting from moving it.
# isort: split

import torch

from verl import DataProto
from verl.trainer import reward_registry
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.apis.ctgov import CTGovAPI
from verl.utils.apis.pubmed import PubmedAPI
from verl.utils.credentials import PUBMED_API_KEY_ENV, load_pubmed_api_key
from verl.utils.logging_utils import configure_logging
from verl.utils.toolchain import require_triton_toolchain

logger = logging.getLogger(__name__)

# The NCBI key comes from $PUBMED_API_KEY (code/.env); this key file (resolved here, not from the cwd) is the fallback.
PUBMED_API_KEY_PATH = os.path.join(project_root, 'verl', 'utils', 'reward_score', 'apis', 'pubmed_api.key')
    
def _select_rm_score_fn(data_source) -> Callable:
    """The compute_score for data_source (see verl/trainer/reward_registry.py); NotImplementedError if none."""
    return reward_registry.select_compute_score(data_source)


class RewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        
        # Initialize APIs globally
        self.pubmed_api = None
        self.ctgov_api = None
        api_key = load_pubmed_api_key(PUBMED_API_KEY_PATH)
        if api_key is None:
            logger.warning('No PubMed API key ($%s unset, no key at %s); using keyless PubMed requests '
                           '(NCBI allows 3 req/s without a key instead of 10).', PUBMED_API_KEY_ENV, PUBMED_API_KEY_PATH)
        self.pubmed_api = PubmedAPI(api_key=api_key)
        self.ctgov_api = CTGovAPI()
        
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        assert data.batch is not None, 'reward input is the rollout batch'
        if 'rm_scores' in data.batch.keys():  # pyright: ignore[reportOperatorIssue]  # tensordict 0.5 types NestedKey as a runtime class, so str keys are rejected by the stub
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        search_apis = reward_registry.SearchApis(pubmed=self.pubmed_api, ctgov=self.ctgov_api)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            assert data_item.batch is not None, 'DataProto.__getitem__ always slices the tensor batch'

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            
            route = reward_registry.require_route(data_source)
            compute_score_fn = route.compute_score_fn(data_source)
            extra_kwargs = route.build_kwargs(data_item.non_tensor_batch, data_source, search_apis)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, **extra_kwargs)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                logger.info('%s', sequences_str)

        return reward_tensor


import hydra
import ray


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    configure_logging()
    # Fail in the first second, not inside step 1: Triton JIT (flash-attn kernels) needs a C compiler and Python.h.
    require_triton_toolchain()
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # GPU placement groups wait forever (no error) when the cluster cannot satisfy them.
    requested_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    available_gpus = ray.cluster_resources().get('GPU', 0)
    if available_gpus < requested_gpus:
        raise RuntimeError(f'trainer.n_gpus_per_node * trainer.nnodes requests {requested_gpus} GPUs '
                           f'but the Ray cluster has {available_gpus:g}; fix the config or CUDA_VISIBLE_DEVICES.')

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    configure_logging()  # Ray workers do not inherit the driver's logging config; rewards log from here
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
