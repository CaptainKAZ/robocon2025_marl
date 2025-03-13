import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env import MultiAgentEnv
from config import Config
from batched_env import BatchedMultiAgentEnv
import ray
from pprint import pprint
from visualizer import BatchRobotVisualizer
import traceback
import pygame
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces.space_utils import batch as batch_func
import tree
from pathlib import Path
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
import os

class FixedRLModule(RLModule):
    @override(RLModule)
    def _forward(self, batch, **kwargs):
        self.fixedAction = (self.action_space.low + self.action_space.high) / 2
        obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
        actions = batch_func([self.fixedAction for _ in range(obs_batch_size)])
        return {SampleBatch.ACTIONS: actions}

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        # RandomRLModule should always be configured as non-trainable.
        # To do so, set in your config:
        # `config.multi_agent(policies_to_train=[list of ModuleIDs to be trained,
        # NOT including the ModuleID of this RLModule])`
        raise NotImplementedError("Random RLModule: Should not be trained!")

    @override(RLModule)
    def output_specs_inference(self):
        return [SampleBatch.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [SampleBatch.ACTIONS]

    def compile(self, *args, **kwargs):
        """Dummy method for compatibility with TorchRLModule.

        This is hit when RolloutWorker tries to compile TorchRLModule."""
        pass


class RLlibWrapperMulti(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.pnum_envs = config.get("num_envs", 1)
        self.visualizer = None
        self.pnum_agents = Config.NUM_AGENTS
        self.env = BatchedMultiAgentEnv(num_envs=self.pnum_envs)
        self._agent_ids = set()
        self.observation_spaces = {}
        self.action_spaces = {}
        for env_index in range(self.pnum_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                self.observation_spaces[agent_id] = self.env.observation_space
                self.action_spaces[agent_id] = self.env.action_space
                self._agent_ids.add(agent_id)

        if not self.agents:
            self.agents = list(self._agent_ids)
        if not self.possible_agents:
            self.possible_agents = self.agents.copy()
        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)
        self.visual_env_idx = None

    def reset(self, seed=None, options=None):
        print("Calling Reset!!!!")
        obs, info = self.env.reset()
        self.visual_env_idx = None
        observations = {
            (i, j): obs[i, j]
            for i in range(self.pnum_envs)
            for j in range(self.pnum_agents)
        }
        pygame.quit()
        self.visualizer = None
        return observations, info
    
    def close(self):
        pygame.quit()
        self.visualizer = None
        return super().close()

    def step(self, action_dict):
        if not all(
            isinstance(agent_id, tuple) and len(agent_id) == 2
            for agent_id in action_dict.keys()
        ):
            raise ValueError(
                "action_dict 的键必须是 (env_index, agent_index) 形式的元组"
            )

        actions = np.zeros(
            (self.pnum_envs, self.pnum_agents, self.env.action_space.shape[0])
        )
        for agent_id, action in action_dict.items():
            env_index, agent_index = agent_id
            actions[env_index][agent_index] = action

        already_done = self.env.get_done_mask()
        obs, rewards, terminated, truncated, _ = self.env.step(actions)

        # 使用 NumPy 向量化操作
        valid_envs = ~already_done
        valid_indices = np.where(valid_envs)[0]
        obs_dict = {
            (i, j): obs[i, j] for i in valid_indices for j in range(self.pnum_agents)
        }
        reward_dict = {
            (i, j): rewards[i, j]
            for i in valid_indices
            for j in range(self.pnum_agents)
        }
        truncated_dict = {
            (i, j): truncated[i, j]
            for i in valid_indices
            for j in range(self.pnum_agents)
        }
        terminated_dict = {
            (i, j): terminated[i, j]
            for i in valid_indices
            for j in range(self.pnum_agents)
        }

        now_done = self.env.get_done_mask()

        try:
            if self.visualizer is None and not pygame.get_init():
                self.visualizer = BatchRobotVisualizer()
            if self.visualizer is not None:
                if self.visual_env_idx is None or now_done[self.visual_env_idx].all():
                    false_indices = np.where(now_done == False)[0]
                    if len(false_indices) >= 4:
                        self.visual_env_idx = np.random.choice(
                            false_indices, size=4, replace=False
                        )
                obs_vis = obs[self.visual_env_idx, :, :]
                positions = obs_vis[:, :, :2]
                velocities = obs_vis[:, :, 2:4]
                targets = obs_vis[:, 0, -2:]
                viz_data = {
                    "positions": positions,  # 形状 [num_envs, 4, 2]
                    "velocities": velocities,  # 形状 [num_envs, 4, 2]
                    "targets": targets,  # 形状 [num_envs, 2]
                    "rewards": rewards[self.visual_env_idx, :],
                }
                # print("calling update")
                self.visualizer.update(viz_data, env_indices=[0, 1, 2, 3])
        except Exception as e:
            print(f"{e}")
            traceback.print_exc()

        terminated_dict["__all__"] = all(terminated_dict.values())
        truncated_dict["__all__"] = all(truncated_dict.values())
        # print(
        #     f"[{self.env.steps}]本轮迭代: {np.sum(now_done)}/{len(now_done)}，{np.sum(now_done)/len(now_done)*100}%"
        # )
        if self.env.steps >= 500:
            terminated_dict["__all__"] = True

        return obs_dict, reward_dict, terminated_dict, truncated_dict, {}


from ray import tune

tune.register_env("custom_env", lambda config: RLlibWrapperMulti(config))

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls

policies = {"attacker1", "attacker2", "defenders"}


def policy_mapping_fn(agent_id, episode, **kwargs):
    (env_index, agent_index) = agent_id
    if agent_index == 0:
        return "attacker1"
    elif agent_index == 1:
        return "attacker2"
    else:
        return "defenders"


parser = add_rllib_example_script_args(
    default_iters=20000,
    default_timesteps=2500000,
    default_reward=100,
)
args = parser.parse_args(
    args=[
        "--algo",
        "PPO",
        "--enable-new-api-stack",
        "--num-env-runners",
        "2",
        # "--no-tune"
        "--num-gpus-per-learner",
        "1",
        "--num-learners",
        "1",
        "--evaluation-num-env-runners",
        "1",
        # "--evaluation-parallel-to-training",
        "--evaluation-interval",
        "5",
        "--evaluation-duration",
        "1",
        "--evaluation-duration-unit",
        "episodes",
        "--num-cpus-per-learner",
        "0",
        # "--local-mode",
        # "--num-gpus",
        # "1",
        "--checkpoint-at-end",
        "--checkpoint-freq",
        "10",
        "--num-cpus",
        "16",
    ]
)
# Tuner.restore(path="/home/neo/ray_results/PPO_2025-03-13_03-44-31", trainable=...)
# /home/neo/ray_results/PPO_2025-03-13_03-44-31/PPO_custom_env_6aa0e_00000_0_2025-03-13_03-44-31/checkpoint_000031
# best_checkpoint_path = "/home/neo/ray_results/APPO_2025-03-12_03-03-21/APPO_custom_env_800ef_00000_0_2025-03-12_03-03-21/checkpoint_000000/"
best_checkpoint_path = "/home/neo/ray_results/PPO_2025-03-13_03-44-31/PPO_custom_env_6aa0e_00000_0_2025-03-13_03-44-31/checkpoint_000031"
def on_algorithm_init(algorithm, **kwargs):

        # algorithm.restore_from_path(best_checkpoint_path,component=(
        #             COMPONENT_LEARNER_GROUP
        #             + "/"
        #             + COMPONENT_LEARNER
        #             + "/"
        #             + COMPONENT_RL_MODULE
        #             + "/attacker1"
        #         ),)
        pass

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .callbacks(on_algorithm_init=on_algorithm_init)
    .environment("custom_env", env_config={"num_envs": 128})
    .multi_agent(
        policies=policies,
        # All agents map to the exact same policy.
        policy_mapping_fn=policy_mapping_fn,
    )
    .training(
        # model={
        #     "vf_share_layers": True,
        # },
        # vf_loss_coeff=0.005,
        num_epochs=5,
        train_batch_size=1000,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "attacker1": RLModuleSpec(
                # .from_module(
                #     RLModule.from_checkpoint(
                #         os.path.join(
                #             "/home/neo/ray_results/PPO_2025-03-13_03-44-31/PPO_custom_env_6aa0e_00000_0_2025-03-13_03-44-31/checkpoint_000031",
                #             "learner_group",
                #             "learner",
                #             "rl_module",
                #             "attacker1",
                #         )
                #     )
                ),
                "attacker2": RLModuleSpec(),
                "defenders": RLModuleSpec(),
                # "fixed_policy": RLModuleSpec(
                #     module_class=RandomRLModule,
                # ),
            },
        ),
    )
)
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_TRAINED,
)
from ray.tune.result import TIME_TOTAL_S, TRAINING_ITERATION

# stop={"training_iteration": 20}
results = run_rllib_example_script_experiment(base_config, args)

module_chkpt_path = (
        Path(results.get_best_result().checkpoint.path)
        / COMPONENT_LEARNER_GROUP
        / COMPONENT_LEARNER
        / COMPONENT_RL_MODULE
        / "attacker1"
    )
assert module_chkpt_path.is_dir()
pprint(module_chkpt_path)
