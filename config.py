import importlib
from pathlib import Path
class Config:
    # 物理参数
    MAX_ACCEL = 3.0  # m/s² (原MAX_SPEED改为MAX_ACCEL)
    MAX_SPEED = 5.0  # 新增速度限制
    ROBOT_RADIUS = 0.3
    DT = 0.1
    FRICTION = 0.0
    COLLISION_DECAY = 0.95

    # 游戏设置
    MAX_TIME = 60

    # 场地参数
    WIDTH = 15
    HEIGHT = 8
    BUFFER = 13

    # 目标参数
    TARGET_X_RANGE = (8, 11)
    TARGET_Y_RANGE = (0.5, 7.5)
    TARGET_VEL_THRESHOLD = 0.3
    # 可视化
    SCREEN_SIZE = (1500, 800)
    SCALE = 100

    # 奖励参数
    R0_DIST_SCALE = 1.5
    R0_DIST_VEC_SCALE = 4.0
    R0_SPEED_PENALTY = 0.02
    R0_SPEED_PENALTY_DIST_THRESHOLD = 2.0
    R1_R0_SCALE = 0.3
    R1_OPPONENT_TARGET_SCALE = 0.3
    R1_OPPONENT_R0_SCALE = 0.1
    R2_R0_SCALE = 0.8
    R2_PROXIMITY_SCALE = 0.3
    R2_R1_SPEED_SCALE = 0.2
    R3_R0_SCALE = 0.8
    R3_PROXIMITY_SCALE = 0.2
    R3_R1_SPEED_SCALE = 0.2
    TIME_REWARD = 0.002

    VIOLATION_PENALTY = 20.0
    DEFENDER_BONUS = 2

    WALL_PENALTY = 6.0

    TEAM_SUCCESS_REWARD = 15.0
    OPPONENT_FAIL_PENALTY = 4.0
    ATTACKER_TIMEOUT_PENALTY = 25
    DEFENDER_TIMEOUT_BONUS = 50

    REWARD_SCALE = 10
    INTER_AGENT_SAFE = 2.0  # 触发机器人间距惩罚的阈值
    INTER_AGENT_SCALE = 1.0  # 基础惩罚系数（最大惩罚值为该系数）
    WALL_SAFE_DISTANCE = 0.5  # 触发离墙惩罚的阈值
    WALL_PROXIMITY_SCALE = 0.5  # 离墙最大惩罚值

    # 系统参数
    NUM_AGENTS = 4
    OBS_DIM = 19  # 2(pos) + 2(vel) + 2(team_pos) + 2(team_vel) + 4(opp_pos) + 4(opp_vel) + 2(target) + 1(time)
    @classmethod
    def reload(cls):
        """动态重新加载配置"""
        print("Reloading configuration...")
        try:
            importlib.reload(importlib.import_module(__name__))
            new_config = importlib.import_module(__name__).Config
            # 更新类属性
            for attr in dir(new_config):
                if not attr.startswith('_'):
                    setattr(cls, attr, getattr(new_config, attr))
            print("Configuration reloaded successfully")
        except Exception as e:
            print(f"Reload failed: {str(e)}")