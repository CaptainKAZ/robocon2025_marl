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


class RLlibWrapper(MultiAgentEnv):
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
        observations = {}
        for env_index in range(self.pnum_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                observations[agent_id] = obs[env_index][agent_index]
        pygame.quit()
        self.visualizer=None
        return observations, info

    def step(self, action_dict):
        actions = np.zeros(
            (self.pnum_envs, self.pnum_agents, self.env.action_space.shape[0])
        )
        for agent_id, action in action_dict.items():
            env_index, agent_index = agent_id
            actions[env_index][agent_index] = action
        obs, rewards, done, info = self.env.step(actions)
        observations = {}
        reward_dict = {}
        done_dict = {}
        truncated_d = {}
        for env_index in range(self.pnum_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                if not done[env_index]:
                    observations[agent_id] = obs[env_index][agent_index]
                    reward_dict[agent_id] = rewards[env_index][agent_index]
                done_dict[agent_id] = done[env_index]
                truncated_d[agent_id] = False
        print(f"本轮迭代：{sum(done_dict.values())}/{len(done_dict)}，{sum(done_dict.values())/len(done_dict)*100}%")
        try:
            if self.visualizer is None:
                self.visualizer = BatchRobotVisualizer()
            if self.visual_env_idx is None or done[self.visual_env_idx].all():
                false_indices = np.where(done == False)[0]
                if len(false_indices) >= 4:
                    self.visual_env_idx = np.random.choice(false_indices, size=4, replace=False)
            obs_vis = obs[self.visual_env_idx,:,:]
            positions = obs_vis[:, :, :2]
            velocities = obs_vis[:, :, 2:4]
            targets = obs_vis[:, 0, -2:]
            viz_data = {
                    "positions": positions,  # 形状 [num_envs, 4, 2]
                    "velocities": velocities,  # 形状 [num_envs, 4, 2]
                    "targets": targets,  # 形状 [num_envs, 2]
                    "rewards": rewards[self.visual_env_idx,:],
            }
            print("calling update")
            self.visualizer.update(viz_data, env_indices=[0,1,2,3])
        except Exception as e:
            print(f"{e}")
            traceback.print_exc()

        done_dict["__all__"] = all(done_dict.values())
        truncated_d["__all__"] = all(truncated_d.values())
        return observations, reward_dict, done_dict, truncated_d, {}


from ray import tune

tune.register_env("custom_env", lambda config: RLlibWrapper(config))

from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy, PPOConfig

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
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)
args = parser.parse_args(
    args=[
        "--enable-new-api-stack",
        "--num-env-runners",
        "1",
        "--num-gpus-per-learner",
        "1",
        "--num-learners",
        "1",
        "--evaluation-num-env-runners",
        "1"
    ]
)
base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment("custom_env", env_config={"num_envs": 1024})
    .multi_agent(
        policies={"attacker1", "attacker2", "defenders"},
        # All agents map to the exact same policy.
        policy_mapping_fn=policy_mapping_fn,
    )
    .training(
        model={
            "vf_share_layers": True,
        },
        vf_loss_coeff=0.005,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "attacker1": RLModuleSpec(),
                "attacker2": RLModuleSpec(),
                "defenders": RLModuleSpec(),
            },
        ),
    )
)
run_rllib_example_script_experiment(base_config, args)
