# config.py
class Config:
    # 物理参数
    MAX_ACCEL = 3.0  # m/s² (原MAX_SPEED改为MAX_ACCEL)
    MAX_SPEED = 5.0  # 新增速度限制
    ROBOT_RADIUS = 0.3
    DT = 0.1
    FRICTION = 0.1

    # 场地参数
    WIDTH = 15
    HEIGHT = 8
    BUFFER = 13

    # 目标参数
    TARGET_X_RANGE = (8, 11)
    TARGET_Y_RANGE = (0.5, 7.5)
    TARGET_VEL_THRESHOLD= 0.1
    # 可视化
    SCREEN_SIZE = (1500, 800)
    SCALE = 100
    
    # 奖励参数
    R0_DIST_SCALE = 0.5
    R0_SPEED_PENALTY = 0.3
    TARGET_THRESHOLD = 2.0
    R1_R0_SCALE = 0.6
    R1_OPPONENT_SCALE = 0.4
    R2_R0_SCALE = 0.6
    R2_PROXIMITY_SCALE = 0.2
    R3_R0_SCALE = 0.6
    R3_PROXIMITY_SCALE = 0.2
    TIME_REWARD = 0.02
    VIOLATION_PENALTY = 1.5
    WALL_PENALTY = 1.0
    TEAM_SUCCESS_REWARD = 8.0
    OPPONENT_FAIL_PENALTY = 4.0
    COLLISION_DECAY = 0.95
    REWARD_SCALE = 0.1
    
    # 系统参数
    NUM_AGENTS = 4
    OBS_DIM = 18  # 2(pos) + 2(vel) + 2(team_pos) + 2(team_vel) + 4(opp_pos) + 4(opp_vel) + 2(target)

