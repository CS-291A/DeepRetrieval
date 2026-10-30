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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  
sys.path.insert(0, project_root)  

import utils.java_init  # noqa: F401  (imported for its side effect: sets JAVA_HOME)

# utils.java_init must run before pyserini is imported; keep import sorting from moving it.
# isort: split

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.apis.ctgov import CTGovAPI
from verl.utils.apis.pubmed import PubmedAPI

logger = logging.getLogger(__name__)

# README: put the NCBI key at code/verl/utils/reward_score/apis/pubmed_api.key (resolved here, not from the cwd)
PUBMED_API_KEY_PATH = os.path.join(project_root, 'verl', 'utils', 'reward_score', 'apis', 'pubmed_api.key')
    
def _select_rm_score_fn(data_source) -> Callable:

    if "screening" in data_source:
        from verl.utils.reward_score import screening
        return screening.compute_score
    elif "pubmed" in data_source:
        from verl.utils.reward_score import pubmed
        return pubmed.compute_score
    elif "pmd_no_reason" in data_source:
        from verl.utils.reward_score import pmd_no_reason
        return pmd_no_reason.compute_score
    elif "ctgov" in data_source:
        from verl.utils.reward_score import ctgov
        return ctgov.compute_score
    elif 'fiqa' in data_source:
        from verl.utils.reward_score import fiqa
        return fiqa.compute_score
    elif 'scifact' in data_source:
        if 'dense' in data_source:
            from verl.utils.reward_score_dense import scifact
            return scifact.compute_score
        else:
            from verl.utils.reward_score import scifact
            return scifact.compute_score
    elif 'nfcorpus' in data_source:
        if 'dense' in data_source:
            from verl.utils.reward_score_dense import nfcorpus
            return nfcorpus.compute_score
        else:
            from verl.utils.reward_score import nfcorpus
            return nfcorpus.compute_score
    elif 'nq_serini' in data_source:
        if 'no_reason' in data_source:
            from verl.utils.reward_score import nq_no_reason
            return nq_no_reason.compute_score
        else:
            from verl.utils.reward_score import nq_serini
            return nq_serini.compute_score
    elif 'triviaqa' in data_source:
        if 'no_reason' in data_source:
            from verl.utils.reward_score import tqa_no_reason
            return tqa_no_reason.compute_score
        else:
            from verl.utils.reward_score import triviaqa
            return triviaqa.compute_score
    elif 'squad' in data_source:
        if 'no_reason' in data_source:
            from verl.utils.reward_score import squad_no_reason
            return squad_no_reason.compute_score
        else:
            from verl.utils.reward_score import squad
        return squad.compute_score
    elif 'hotpotqa' in data_source:
        if 'dense' in data_source:
            from verl.utils.reward_score_dense import hotpotqa
            return hotpotqa.compute_score
        else:
            from verl.utils.reward_score import hotpotqa
            return hotpotqa.compute_score
    elif 'fever' in data_source:
        if 'dense' in data_source:
            from verl.utils.reward_score_dense import fever
            return fever.compute_score
        else:
            from verl.utils.reward_score import fever
            return fever.compute_score
    elif 'msmarco_beir' in data_source:
        if 'dense' in data_source:
            from verl.utils.reward_score_dense import msmarco_beir
            return msmarco_beir.compute_score
        else:
            from verl.utils.reward_score import msmarco_beir
            return msmarco_beir.compute_score
    elif 'msmarco' in data_source:
        from verl.utils.reward_score import msmarco
        return msmarco.compute_score
    elif 'bird' in data_source:
        if 'wor' in data_source:
            from verl.utils.reward_score import bird_no_reason
            return bird_no_reason.compute_score
        else:
            from verl.utils.reward_score import bird
            return bird.compute_score
    elif 'spider' in data_source:
        if 'wor' in data_source:
            from verl.utils.reward_score import spider_no_reason
            return spider_no_reason.compute_score
        else:
            from verl.utils.reward_score import spider
            return spider.compute_score
    else:
        raise NotImplementedError(f"No reward function for data_source '{data_source}'")


class RewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        
        # Initialize APIs globally
        self.pubmed_api = None
        self.ctgov_api = None
        api_key = None
        if os.path.exists(PUBMED_API_KEY_PATH):
            with open(PUBMED_API_KEY_PATH, 'r') as f:
                api_key = f.read().strip()
        else:
            logger.warning(f'PubMed API key not found at {PUBMED_API_KEY_PATH}; using keyless PubMed requests '
                           '(NCBI allows 3 req/s without a key instead of 10).')
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
            
            compute_score_fn = _select_rm_score_fn(data_source)

            if 'pubmed' in data_source or 'ctgov' in data_source or 'pmd_no_reason' in data_source:
                pub_date = data_item.non_tensor_batch['pub_date']
                literature_type = 'publication' if ('pubmed' in data_source or 'pmd_no_reason' in data_source) else 'trial'
                if literature_type == 'publication':
                    api = self.pubmed_api
                elif literature_type == 'trial':
                    api = self.ctgov_api
                else:
                    raise ValueError('Invalid literature type.')
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, search_api=api, literature_type=literature_type, pub_date=pub_date)
            elif 'scifact' in data_source or 'fiqa' in data_source or 'nfcorpus' in data_source or 'hotpotqa' in data_source \
                    or 'fever' in data_source or 'msmarco' in data_source:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, data_source=data_source)
            elif 'bird' in data_source or 'spider' in data_source:
                db_path = data_item.non_tensor_batch['extra_info']['db_path']
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, data_source=data_source, db_path=db_path)
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import hydra
import ray


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
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
