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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from agent.tool import ToolEnv
from agent.tool.tools import _default_tools

import ray
import hydra

from verl import DataProto
from verl.utils.reward_score import _default_compute_score_format, _default_compute_score_answer_f1, _default_compute_score_answer_em, _default_compute_score_format_answer
import torch

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        answer_lst_f1 = []
        answer_lst_em = []
        format_lst = []
        result_lst = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            valid_response_ids = response_ids[:valid_response_length].long()

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']

            score = _default_compute_score_format_answer(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            answer_f1_score = _default_compute_score_answer_f1(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            answer_em_score = _default_compute_score_answer_em(data_source=data_source, solution_str=sequences_str, ground_truth=ground_truth)
            format_score = _default_compute_score_format(data_source=data_source, solution_str=sequences_str)

            answer_lst_f1.append(answer_f1_score)
            answer_lst_em.append(answer_em_score)
            format_lst.append(format_score)
            result_lst.append(sequences_str)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return reward_tensor, answer_lst_f1, answer_lst_em, format_lst, result_lst
    



@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    # ray.init(runtime_env={"env_vars": {"RAY_DEBUG_POST_MORTEM": "1"}})
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    # breakpoint()
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
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

    # reward_manager_name = config.reward_model.get("reward_manager", "naive")
    # if reward_manager_name == 'naive':
    #     from verl.workers.reward_manager import NaiveRewardManager
    #     reward_manager_cls = NaiveRewardManager
    # elif reward_manager_name == 'prime':
    #     from verl.workers.reward_manager import PrimeRewardManager
    #     reward_manager_cls = PrimeRewardManager
    # else:
    #     raise NotImplementedError
    # reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    # val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    tools = _default_tools(config.tool.env)
    env = ToolEnv(tools=tools, max_turns=config.tool.max_turns)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=RewardManager(tokenizer=tokenizer, num_examine=0),
                            val_reward_fn=RewardManager(tokenizer=tokenizer, num_examine=1),
                            env=env)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
