import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env import MultiAgentEnv
from config import Config
from robocon2025_marl.batched_env import BatchedMultiAgentEnv


class RLlibWrapper(MultiAgentEnv):
    def __init__(self, config):
        self.num_envs = config.get("num_envs", 1)
        self.pnum_agents = Config.NUM_AGENTS
        self.env = BatchedMultiAgentEnv(num_envs=self.num_envs)

        self.observation_space = {}
        self.action_space = {}
        for env_index in range(self.num_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                self.observation_space[agent_id] = self.env.observation_space
                self.action_space[agent_id] = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        observations = {}
        for env_index in range(self.num_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                observations[agent_id] = obs[env_index][agent_index]
        return observations, info

    def step(self, action_dict):
        actions = np.zeros(
            (self.num_envs, self.pnum_agents, self.env.action_space.shape[0])
        )
        for agent_id, action in action_dict.items():
            env_index, agent_index = agent_id
            actions[env_index][agent_index] = action
        obs, rewards, done, info = self.env.step(actions)
        observations = {}
        reward_dict = {}
        done_dict = {}
        for env_index in range(self.num_envs):
            for agent_index in range(self.pnum_agents):
                agent_id = (env_index, agent_index)
                observations[agent_id] = obs[env_index][agent_index]
                reward_dict[agent_id] = rewards[env_index][agent_index]
                done_dict[agent_id] = done[env_index]
        return observations, reward_dict, done_dict, info


class CustomMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config):
        self.wrapper = RLlibWrapper(config)

    def reset(self):
        return self.wrapper.reset()

    def step(self, action_dict):
        return self.wrapper.step(action_dict)


from ray import tune

tune.register_env("custom_env", lambda config: CustomMultiAgentEnv(config))

from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy, PPOConfig

# 定义观测和动作空间
obs_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(Config.OBS_DIM,), dtype=np.float32
)
action_space = spaces.Box(
    low=-Config.MAX_ACCEL, high=Config.MAX_ACCEL, shape=(2,), dtype=np.float32
)

config = (
    PPOConfig()
    .environment("custom_env", env_config={"num_envs": 256})
    .env_runners(num_env_runners=1)
    .framework("torch")
    .learners(
        num_learners=1,
        num_gpus_per_learner=1,
    )
    .multi_agent(
        policies={
            "team1_policy": (
                PPOTorchPolicy,
                {},
                {"observation_space": obs_space, "action_space": action_space},
                {},
            ),
            "team2_policy": (
                PPOTorchPolicy,
                {},
                {"observation_space": obs_space, "action_space": action_space},
                {},
            ),
        },
        policy_mapping_fn=lambda agent_id: (
            "team1_policy" if agent_id[1] < 2 else "team2_policy"
        ),
    )
)

trainer = PPO(config)

# 训练
num_iterations = 100
for i in range(num_iterations):
    result = trainer.train()
    print(f"Iteration {i}, Reward: {result['episode_reward_mean']}")
