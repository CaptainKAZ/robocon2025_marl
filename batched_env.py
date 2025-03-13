import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config

class BatchedMultiAgentEnv(gym.Env):
    def __init__(self, num_envs=1024):
        super().__init__()

        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化状态张量
        self.positions = torch.zeros(
            (num_envs, Config.NUM_AGENTS, 2), dtype=torch.float32, device=self.device
        )
        self.velocities = torch.zeros_like(self.positions)
        self.targets = torch.zeros(
            (num_envs, 2), dtype=torch.float32, device=self.device
        )

        # 碰撞记录
        self.collision_speeds = torch.zeros_like(self.positions)
        self.boundary_collision_flags = torch.zeros(
            (num_envs, Config.NUM_AGENTS), dtype=torch.bool, device=self.device
        )

        # 定义空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(Config.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-Config.MAX_ACCEL, high=Config.MAX_ACCEL, shape=(2,), dtype=np.float32
        )

        self.done_mask = torch.zeros(
            (num_envs,), dtype=torch.bool, device=self.device
        )
        self.steps = 0

    def get_done_mask(self):
        """获取done_mask"""
        return self.done_mask.cpu().numpy()

    def _physics_step(self, accelerations):
        """执行物理模拟步骤"""
        # 限制加速度
        acc_norms = torch.norm(accelerations, dim=2, keepdim=True)
        accelerations = torch.where(
            acc_norms > Config.MAX_ACCEL,
            accelerations / (acc_norms + 1e-6) * Config.MAX_ACCEL,
            accelerations,
        )

        # 更新速度
        self.velocities += accelerations * Config.DT

        # 限制速度
        speed_norms = torch.norm(self.velocities, dim=2, keepdim=True)
        self.velocities = torch.where(
            speed_norms > Config.MAX_SPEED,
            self.velocities / speed_norms * Config.MAX_SPEED,
            self.velocities,
        )

        # 记录原始速度用于碰撞检测
        original_velocities = self.velocities.clone()

        # 更新位置并检测边界碰撞
        new_positions = self.positions + self.velocities * Config.DT
        clamped_pos = torch.clamp(
            new_positions,
            min=torch.tensor([0.0, 0.0], device=self.device),
            max=torch.tensor([Config.WIDTH, Config.HEIGHT], device=self.device),
        )

        # 检测碰撞并记录速度
        pos_diff = clamped_pos != new_positions
        self.boundary_collision_flags = pos_diff.any(dim=2)
        self.collision_speeds = torch.where(
            self.boundary_collision_flags.unsqueeze(-1),
            original_velocities,
            torch.zeros_like(original_velocities),
        )

        # 应用位置修正和速度清零
        self.positions = clamped_pos
        self.velocities[self.boundary_collision_flags] = 0.0

        # 碰撞速度衰减
        self.collision_speeds *= Config.COLLISION_DECAY

    def step(self, actions):
        """环境步进"""
        # 转换动作到tensor [num_envs, agents, action_dim]
        already_done = self.done_mask.clone()
        accelerations = torch.as_tensor(
            actions, device=self.device, dtype=torch.float32
        )
        accelerations = accelerations.reshape(self.num_envs, Config.NUM_AGENTS, -1)
        accelerations[already_done] = 0.0
        # 物理模拟
        self._physics_step(accelerations)

        # 碰撞检测
        collision_matrix, violation_flags, distances = self._check_violation()
        collision_dones = collision_matrix.any(dim=(1, 2))

        # 计算奖励
        rewards = self._compute_base_rewards()
        rewards = self._handle_collision_penalties(rewards, violation_flags)
        rewards = self._handle_boundary_penalties(rewards)
        rewards = self._handle_agent_proximity_penalties(rewards, distances)
        rewards = self._handle_wall_proximity_penalties(rewards)
        # 终止条件和奖励计算
        reach_target, rewards = self._compute_terminal_rewards(rewards, collision_dones)
        # 奖励缩放
        rewards *= Config.REWARD_SCALE

        # terminated 和 truncated 的计算
        collision_any = collision_matrix.any(dim=(1, 2))  # [num_envs]
        agent_collisions = collision_matrix.any(dim=2)  # [num_envs, NUM_AGENTS]

        # terminated: 如果智能体发生碰撞，则为 True
        terminated = torch.zeros(
            (self.num_envs, Config.NUM_AGENTS), dtype=torch.bool, device=self.device
        )
        terminated |= agent_collisions.bool()  # 碰撞导致终止
        terminated[reach_target] = True  # agent0 到达目标点导致终止

        # truncated: 如果有碰撞但智能体未直接参与碰撞，则为 True
        truncated = (
            collision_any.unsqueeze(1) & ~agent_collisions
        ).bool()  # [num_envs, NUM_AGENTS]

        # # 超时
        # if self.steps > Config.MAX_TIME / Config.DT:
        #     truncated = ~terminated
        self.steps += 1

        self.done_mask |= collision_dones | reach_target
        self.velocities[already_done] = 0.0
        rewards[already_done] = 0.0

        obs = self._get_obs()

        info = {
            "violations": violation_flags.cpu().numpy(),
            "collisions": collision_matrix.cpu().numpy(),
        }

        return obs, rewards.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info

    def _compute_base_rewards(self):
        """计算基础奖励分量"""
        rewards = torch.zeros((self.num_envs, Config.NUM_AGENTS), device=self.device)

        # 公共计算
        r0_pos = self.positions[:, 0]
        r0_dist = torch.norm(r0_pos - self.targets, dim=1)
        r0_speed = torch.norm(self.velocities[:, 0], dim=1)
        opp_positions = self.positions[:, 2:]  # 所有对手的位置（R2和R3）

        if self.r0_last_dist is None:
            self.r0_last_dist = r0_dist

        opp_positions = self.positions[:, 2:]
        opp_dist = torch.norm(r0_pos.unsqueeze(1) - opp_positions, dim=2)

        # R0奖励：接近目标 + 末端减速
        speed_penalty = torch.where(
            r0_dist < Config.TARGET_THRESHOLD,
            torch.norm(self.velocities[:, 0], dim=1) * Config.R0_SPEED_PENALTY,
            0.0,
        )

        rewards[:, 0] = (-r0_dist * Config.R0_DIST_SCALE) + ((self.r0_last_dist - r0_dist) * Config.R0_DIST_VEC_SCALE) - speed_penalty - Config.TIME_REWARD * self.steps

        self.r0_last_dist = r0_dist

        # R1奖励：支持R0 + 保持距离
        opp_to_target_dist = torch.norm(opp_positions - self.targets.unsqueeze(1), dim=2)  # [num_envs, 2]
        reward_opp_away_from_target = opp_to_target_dist.mean(dim=1) * Config.R1_OPPONENT_TARGET_SCALE  # 对手离目标越远奖励越高

        # 2. 驱赶对手远离R0
        opp_to_r0_dist = torch.norm(opp_positions - r0_pos.unsqueeze(1), dim=2)  # [num_envs, 2]
        reward_opp_away_from_r0 = opp_to_r0_dist.mean(dim=1) * Config.R1_OPPONENT_R0_SCALE  # 对手离R0越远奖励越高

        # 合并奖励项
        rewards[:, 1] = (
            (-r0_dist * Config.R1_R0_SCALE)  # 原有：鼓励R0接近目标
            + reward_opp_away_from_target    # 新增：对手远离目标
            + reward_opp_away_from_r0         # 新增：对手远离R0
        )

        # R2/R3奖励：阻止R0 + 靠近R0
        rewards[:, 2] = (
            (r0_dist * Config.R2_R0_SCALE)
            + (1 / (opp_dist[:, 0] + 1e-6)) * Config.R2_PROXIMITY_SCALE
            + Config.TIME_REWARD*self.steps + r0_speed * Config.R2_R1_SPEED_SCALE
        )
        rewards[:, 3] = (
            (r0_dist * Config.R3_R0_SCALE)
            + (1 / (opp_dist[:, 1] + 1e-6)) * Config.R3_PROXIMITY_SCALE
            + Config.TIME_REWARD*self.steps + r0_speed * Config.R2_R1_SPEED_SCALE
        )

        return rewards

    def _handle_agent_proximity_penalties(self, rewards, distances):
        """动态机器人间距惩罚（基于距离倒数）"""

        # 生成掩码并计算惩罚量
        eye_mask = torch.eye(Config.NUM_AGENTS, device=self.device).bool().unsqueeze(0)
        safe_mask = distances < Config.INTER_AGENT_SAFE

        # 动态惩罚计算：距离越近惩罚越大
        penalty_ratio = (Config.INTER_AGENT_SAFE - distances) / Config.INTER_AGENT_SAFE
        penalty_ratio = torch.clamp(penalty_ratio, min=0)  # 负值归零
        penalties = penalty_ratio * Config.INTER_AGENT_SCALE

        # 排除自身并聚合惩罚
        total_penalty = torch.where(eye_mask | ~safe_mask, 0.0, penalties).sum(dim=2)
        rewards -= total_penalty

        return rewards

    def _handle_wall_proximity_penalties(self, rewards):
        """动态离墙距离惩罚（基于距离倒数）"""
        positions = self.positions  # [n_env, n_agent, 2]

        # 计算到各边界的距离
        dist_left = positions[..., 0]
        dist_right = Config.WIDTH - positions[..., 0]
        dist_bottom = positions[..., 1]
        dist_top = Config.HEIGHT - positions[..., 1]

        # 获取到最近墙体的距离
        min_wall_dist = torch.stack([
            dist_left, 
            dist_right,
            dist_bottom,
            dist_top
        ], dim=-1).min(dim=-1)[0]  # [n_env, n_agent]

        # 动态惩罚计算：距离越近惩罚越大
        safe_mask = min_wall_dist < Config.WALL_SAFE_DISTANCE
        penalty_ratio = (Config.WALL_SAFE_DISTANCE - min_wall_dist) / Config.WALL_SAFE_DISTANCE
        penalty_ratio = torch.clamp(penalty_ratio, min=0)
        penalties = penalty_ratio * Config.WALL_PROXIMITY_SCALE

        rewards -= torch.where(safe_mask, penalties, 0.0)
        return rewards

    def _handle_collision_penalties(self, rewards, violation_flags):
        """处理碰撞违规惩罚"""
        # 队伍掩码
        team_masks = torch.zeros(
            (2, Config.NUM_AGENTS), dtype=torch.bool, device=self.device
        )
        team_masks[0, :2] = True  # 队伍1 (0,1)
        team_masks[1, 2:] = True  # 队伍2 (2,3)

        # 检测队伍违规
        team_violations = torch.stack(
            [(violation_flags & mask).any(dim=1, keepdim=True) for mask in team_masks]
        )

        # 清零违规队伍奖励
        for i, mask in enumerate(team_masks):
            rewards[:, mask] *= ~team_violations[i]

        # 违规者额外惩罚
        rewards -= violation_flags.float() * Config.VIOLATION_PENALTY
        return rewards

    def _handle_boundary_penalties(self, rewards):
        """处理边界碰撞惩罚"""
        collision_speed = torch.norm(self.collision_speeds, dim=2)
        boundary_penalty = (
            self.boundary_collision_flags.float()
            * collision_speed
            * Config.WALL_PENALTY
        )
        return rewards - boundary_penalty * Config.DT

    def _compute_terminal_rewards(self, rewards, collision_dones):
        """计算终止条件奖励"""
        # 到达目标检测
        r0_dist = torch.norm(self.positions[:, 0] - self.targets, dim=1)
        reach_target = (r0_dist < Config.ROBOT_RADIUS) & (
            torch.norm(self.velocities[:, 0], dim=1) < Config.TARGET_VEL_THRESHOLD
        )

        # 队伍奖励/惩罚
        rewards[:, [0, 1]] += reach_target.unsqueeze(1) * (Config.TEAM_SUCCESS_REWARD)
        rewards[:, [2, 3]] -= reach_target.unsqueeze(1) * Config.OPPONENT_FAIL_PENALTY

        return reach_target, rewards

    def _get_obs(self):
        """构建观测张量"""
        obs = torch.zeros(
            (self.num_envs, Config.NUM_AGENTS, Config.OBS_DIM), 
            device=self.device,
            dtype=torch.float32
        )

        for agent_id in range(Config.NUM_AGENTS):
            # 自身状态
            self_pos = self.positions[:, agent_id]
            self_vel = self.velocities[:, agent_id]

            # 队友信息
            teammate_id = 1 if agent_id == 0 else 0 if agent_id == 1 else 3 if agent_id == 2 else 2
            team_pos = self.positions[:, teammate_id]
            team_vel = self.velocities[:, teammate_id]

            # 对手信息
            opp_ids = [2, 3] if agent_id < 2 else [0, 1]
            opp_pos = self.positions[:, opp_ids].flatten(start_dim=1)
            opp_vel = self.velocities[:, opp_ids].flatten(start_dim=1)
            unknown_target = torch.zeros_like(self.targets)

            # 拼接观测
            obs[:, agent_id] = torch.cat(
                [
                    self_pos,
                    self_vel,
                    team_pos,
                    team_vel,
                    opp_pos,
                    opp_vel,
                    self.targets if agent_id < 2 else unknown_target,  # R2 R3不知道R1去哪里投篮
                ],
                dim=1,
            )

        return obs.cpu().numpy()

    def reset(self):
        """带安全检测的环境重置"""
        # 位置初始化（新方法）
        left_x = (Config.WIDTH - Config.BUFFER) / 2
        self._init_positions(
            x_ranges=[(0.0, left_x), (left_x + Config.BUFFER, Config.WIDTH)],
            y_range=(0.0, Config.HEIGHT),
        )
        # 目标点初始化
        self.targets[:, 0] = (
            torch.rand(self.num_envs, device=self.device)
            * (Config.TARGET_X_RANGE[1] - Config.TARGET_X_RANGE[0])
            + Config.TARGET_X_RANGE[0]
        )
        self.targets[:, 1] = (
            torch.rand(self.num_envs, device=self.device)
            * (Config.TARGET_Y_RANGE[1] - Config.TARGET_Y_RANGE[0])
            + Config.TARGET_Y_RANGE[0]
        )

        # 重置速度
        self.velocities.zero_()
        self.collision_speeds.zero_()
        self.boundary_collision_flags.zero_()
        self.done_mask.zero_()
        self.steps = 0
        self.r0_last_dist = None

        return self._get_obs(), {}

    def _init_positions(self, x_ranges, y_range):
        """带碰撞检测的简化位置初始化"""
        # 配置参数
        min_dist = 2 * Config.ROBOT_RADIUS + 0.1  # 最小间距
        max_attempts = 10  # 最大尝试次数

        # 初始化所有智能体位置
        for env_idx in range(self.num_envs):
            for attempt in range(max_attempts):
                # 生成候选位置
                positions = torch.zeros((Config.NUM_AGENTS, 2), device=self.device)
                for agent_idx in range(Config.NUM_AGENTS):
                    team_idx = 0 if agent_idx < 2 else 1  # 队伍索引
                    x_min, x_max = x_ranges[team_idx]
                    positions[agent_idx, 0] = (
                        torch.rand(1, device=self.device) * (x_max - x_min) + x_min
                    )
                    positions[agent_idx, 1] = (
                        torch.rand(1, device=self.device) * (y_range[1] - y_range[0])
                        + y_range[0]
                    )

                # 检查碰撞
                collision = False
                for i in range(Config.NUM_AGENTS):
                    for j in range(i + 1, Config.NUM_AGENTS):
                        dist = torch.norm(positions[i] - positions[j])
                        if dist < min_dist:
                            collision = True
                            break
                    if collision:
                        break

                # 如果没有碰撞，保存位置并退出尝试
                if not collision:
                    self.positions[env_idx] = positions
                    break

            # 如果达到最大尝试次数仍未找到合法位置，强制分散
            if collision:
                print("try fucking so hard!!!")
                angle = torch.linspace(
                    0, 2 * np.pi, Config.NUM_AGENTS, device=self.device
                )
                radius = min_dist * 1.5  # 确保间距
                center_x = (x_ranges[0][1] + x_ranges[1][0]) / 2
                center_y = (y_range[0] + y_range[1]) / 2
                self.positions[env_idx, :, 0] = center_x + radius * torch.cos(angle)
                self.positions[env_idx, :, 1] = center_y + radius * torch.sin(angle)

    def _detect_collisions(self):
        """检测机器人间碰撞"""
        pos_diff = self.positions.unsqueeze(2) - self.positions.unsqueeze(1)
        distances = torch.norm(pos_diff, dim=3)
        eye_mask = torch.eye(
            Config.NUM_AGENTS, device=self.device, dtype=torch.bool
        ).unsqueeze(0)
        return (distances < 2 * Config.ROBOT_RADIUS) & ~eye_mask, distances

    def _check_violation(self):
        """检测碰撞违规"""
        collision_matrix, distances = self._detect_collisions()
        speeds = torch.norm(self.velocities, dim=2)

        # 速度比较矩阵 [num_envs, i, j]
        speed_diff = speeds.unsqueeze(2) - speeds.unsqueeze(1)
        violation_matrix = collision_matrix & (speed_diff >= 0)

        # 聚合违规标记
        violation_flags = violation_matrix.any(dim=2)
        return collision_matrix, violation_flags, distances
