import pygame
import torch
import numpy as np
from batched_env import BatchedMultiAgentEnv
from config import Config
from visualizer import BatchRobotVisualizer


class ManualController:
    def __init__(self, num_envs=4):
        self.env = BatchedMultiAgentEnv(num_envs=num_envs)
        self.visualizer = BatchRobotVisualizer(num_envs_to_show=num_envs)

        # 控制状态
        self.controlled_env = 0
        self.control_mode = "attack"  # attack/defense
        self.clock = pygame.time.Clock()

        # 键位映射（支持多环境控制）
        self.KEY_ACTIONS = {
            pygame.K_w: ("attack", 0, [0.0, 1.0]),
            pygame.K_s: ("attack", 0, [0.0, -1.0]),
            pygame.K_a: ("attack", 0, [-1.0, 0.0]),
            pygame.K_d: ("attack", 0, [1.0, 0.0]),
            pygame.K_UP: ("defense", 2, [0.0, 1.0]),
            pygame.K_DOWN: ("defense", 2, [0.0, -1.0]),
            pygame.K_LEFT: ("defense", 2, [-1.0, 0.0]),
            pygame.K_RIGHT: ("defense", 2, [1.0, 0.0]),
            pygame.K_TAB: ("switch_env", None, None),
            pygame.K_r: ("reset", None, None),
            pygame.K_q: ("quit", None, None),
        }

    def create_actions(self, pressed_keys):
        """生成批量动作张量"""
        actions = torch.zeros((self.env.num_envs, 4, 2), dtype=torch.float32)

        # 遍历所有定义的按键
        for key in self.KEY_ACTIONS:
            if pressed_keys[key]:  # 检查按键是否被按下
                action_type, agent_id, direction = self.KEY_ACTIONS[key]

                # 仅处理攻击/防御模式下的动作
                if action_type in ["attack", "defense"]:
                    # 将方向向量乘以加速度系数
                    actions[self.controlled_env, agent_id] += (
                        torch.tensor(direction, dtype=torch.float32) * Config.MAX_ACCEL
                    )

        # 限制动作范围
        return torch.clamp(actions, -Config.MAX_ACCEL, Config.MAX_ACCEL)

    def process_events(self):
        """处理系统事件"""
        pressed = pygame.key.get_pressed()
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.controlled_env = (self.controlled_env + 1) % self.env.num_envs
                    action_taken = True
                elif event.key == pygame.K_r:
                    self.env.reset()
                    action_taken = True
                elif event.key == pygame.K_q:
                    return "quit"

        return "continue" if not action_taken else "update"

    def run(self):
        """主控制循环"""
        state = self.env.reset()
        running = True

        while running:
            # 处理输入
            action_result = self.process_events()
            if action_result == "quit":
                running = False

            # 生成动作
            pressed = pygame.key.get_pressed()
            actions = self.create_actions(pressed)

            # 环境步进
            with torch.no_grad():
                obs, rewards, dones, info = self.env.step(actions.numpy())

            # 提取所有智能体的位置 (前2维)
            positions = obs[:, :, :2]  # [num_envs, num_agents, 2]

            # 提取所有智能体的速度 (第3-4维)
            velocities = obs[:, :, 2:4]  # [num_envs, num_agents, 2]

            # 提取目标位置 (假设每个智能体的最后2维是目标位置，且所有智能体的目标相同)
            targets = obs[:, 0, -2:]  # [num_envs, 2]

            # 构建可视化数据
            viz_data = {
                "positions": positions,  # 形状 [num_envs, 4, 2]
                "velocities": velocities,  # 形状 [num_envs, 4, 2]
                "targets": targets,  # 形状 [num_envs, 2]
                "violations": info["violations"],  # 形状 [num_envs, 4]
                "dones": dones,  # 形状 [num_envs]
                "rewards": rewards,
            }
            env_indices = list(range(self.env.num_envs))
            env_indices.remove(self.controlled_env)
            env_indices.insert(0, self.controlled_env)
            # 更新显示
            self.visualizer.update(viz_data, env_indices=env_indices)
            self.clock.tick(1 / Config.DT)

        self.visualizer.close()


if __name__ == "__main__":
    controller = ManualController(num_envs=4)
    controller.run()
