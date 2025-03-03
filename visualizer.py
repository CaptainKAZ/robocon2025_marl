# visualizer.py
import pygame
import numpy as np
from config import Config


class BatchRobotVisualizer:
    def __init__(self, num_envs_to_show=4):
        pygame.init()
        self.cfg = Config()

        # 多环境显示布局
        self.num_envs = num_envs_to_show
        self.grid_size = int(np.ceil(np.sqrt(num_envs_to_show)))
        self.cell_size = (
            int(self.cfg.SCREEN_SIZE[0] / self.grid_size),
            int(self.cfg.SCREEN_SIZE[1] / self.grid_size),
        )

        # 主显示表面
        self.screen = pygame.display.set_mode(self.cfg.SCREEN_SIZE)
        pygame.display.set_caption(f"Batch Robot Gym ({num_envs_to_show} Envs)")

        # 颜色和字体配置
        self.colors = [(231, 76, 60), (230, 126, 34), (52, 152, 219), (155, 89, 182)]
        self.font = pygame.font.Font(None, 20)
        self.env_font = pygame.font.Font(None, 14)

    def _world_to_screen(self, pos, env_idx):
        """修正坐标转换逻辑"""
        # 计算行列位置
        col = env_idx % self.grid_size
        row = env_idx // self.grid_size

        # 计算每个环境的绘制区域
        offset_x = col * self.cell_size[0]
        offset_y = row * self.cell_size[1]

        # 添加边界边距防止重叠
        margin = 5
        scale = min(
            (self.cell_size[0] - 2 * margin) / Config.WIDTH,
            (self.cell_size[1] - 2 * margin) / Config.HEIGHT,
        )

        # 精确坐标映射
        return (
            offset_x + margin + int(pos[0] * scale),
            offset_y + margin + int((Config.HEIGHT - pos[1]) * scale),
        )

    def update(self, batch_data, env_indices=None):
        """
        输入参数:
        - batch_data: 包含以下键的字典:
            positions: [num_envs, 4, 2] numpy数组
            violations: [num_envs, 4] bool数组
            velocities: [num_envs, 4, 2] numpy数组 (可选)
        - env_indices: 要显示的环境索引列表
        """
        self.screen.fill((245, 245, 245))

        # 处理输入数据
        positions = batch_data["positions"]
        violations = batch_data.get("violations", np.zeros_like(positions[:, :, 0]))
        velocities = batch_data.get("velocities", None)
        targets = batch_data.get("targets", None)
        dones = batch_data.get("dones",None)
        rewards = batch_data.get("rewards", None)
        # 确定要显示的环境
        if env_indices is None:
            # 显示前N个环境（N=num_envs_to_show）
            env_indices = list(range(self.num_envs))
        else:
            # 限制最大显示数量
            env_indices = env_indices[: self.num_envs]

        # 修正3：绘制每个环境的边界框
        for i, env_idx in enumerate(env_indices):
            # 计算每个环境的绘制区域
            col = env_idx % self.grid_size
            row = env_idx // self.grid_size
            offset = (col * self.cell_size[0], row * self.cell_size[1])

            # 绘制环境边界
            pygame.draw.rect(
                self.screen,
                (0, 0, 255) if i == 0 else (200, 200, 200),  # 高亮第一个环境
                (offset[0], offset[1], self.cell_size[0], self.cell_size[1]),
                2,
            )

            # 绘制环境内容
            self._draw_single_env(
                positions[env_idx],
                violations[env_idx],
                velocities[env_idx] if velocities is not None else None,
                targets[env_idx] if targets is not None else None,  # 新增targets参数
                dones[env_idx],
                rewards[env_idx],
                env_idx,
            )

        # 绘制全局信息
        self._draw_global_info(batch_data)
        pygame.display.flip()

    def _draw_single_env(self, positions, violations, velocities, target, done, rewards, env_idx):
        """绘制单个环境实例"""
        # 绘制边界框
        col = env_idx % self.grid_size
        row = env_idx // self.grid_size
        offset = (col * self.cell_size[0], row * self.cell_size[1])
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (offset[0], offset[1], self.cell_size[0], self.cell_size[1]),
            1,
        )
        if not done:
            # 绘制所有机器人
            for robot_idx in range(4):
                screen_pos = self._world_to_screen(positions[robot_idx], env_idx)
                radius = int(self.cfg.ROBOT_RADIUS * self.cfg.SCALE / self.grid_size)

                # 机器人主体
                pygame.draw.circle(self.screen, self.colors[robot_idx], screen_pos, radius)

                # 违规标记
                if violations[robot_idx]:
                    pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, radius // 2, 2)

                # 速度箭头
                if velocities is not None:
                    vx, vy = velocities[robot_idx]
                    if np.linalg.norm([vx, vy]) > 0.1:
                        end_pos = (
                            screen_pos[0] + int(vx * self.cfg.SCALE * 0.5),
                            screen_pos[1] - int(vy * self.cfg.SCALE * 0.5),
                        )
                        pygame.draw.line(
                            self.screen, (46, 204, 113), screen_pos, end_pos, 2
                        )
                # reward 显示
                if rewards is not None:
                    text = self.env_font.render(f"{rewards[robot_idx]}", True, (100, 100, 100))
                    self.screen.blit(text, (screen_pos[0], screen_pos[1]))

            if target is not None:
                screen_target = self._world_to_screen(target, env_idx)
                cross_size = 8  # 叉叉尺寸
                # 绘制两条交叉线
                pygame.draw.line(
                    self.screen, 
                    (255, 0, 0),  # 红色叉叉
                    (screen_target[0] - cross_size, screen_target[1] - cross_size),
                    (screen_target[0] + cross_size, screen_target[1] + cross_size),
                    2  # 线宽
                )
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (screen_target[0] - cross_size, screen_target[1] + cross_size),
                    (screen_target[0] + cross_size, screen_target[1] - cross_size),
                    2
                )
        # 环境索引标签
        text = self.env_font.render(f"Env {env_idx} Done: {done}", True, (100, 100, 100))
        self.screen.blit(text, (offset[0] + 5, offset[1] + 5))

    def _draw_global_info(self, data):
        """绘制全局统计信息"""
        text_lines = [
            f"Total Envs: {data['positions'].shape[0]}",
            f"Avg Speed: {np.mean(np.linalg.norm(data.get('velocities',0), axis=-1)):.2f}",
            f"Violation Rate: {np.mean(data.get('violations',0)):.2%}",
        ]

        y_pos = 10
        for line in text_lines:
            text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (self.cfg.SCREEN_SIZE[0] - 200, y_pos))
            y_pos += 24

    def close(self):
        pygame.quit()
