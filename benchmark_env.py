import time
import numpy as np
from batched_env import BatchedMultiAgentEnv 
import torch
from config import Config

def benchmark(num_envs=1024, num_steps=1000, warmup=10):
    # 初始化环境
    env = BatchedMultiAgentEnv(num_envs=num_envs)
    
    # 生成测试用动作（直接在GPU上创建）
    actions = torch.rand((num_envs, 4, 2), 
                       device=env.device,
                       dtype=torch.float16) * Config.MAX_ACCEL * 2 - Config.MAX_ACCEL
    
    # 预热阶段
    print("Warming up...")
    for _ in range(warmup):
        env.step(actions)
    
    # 同步GPU确保准确计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 正式测试
    print("Benchmarking...")
    total_time = 0
    physics_time = 0
    collision_time = 0
    reward_time = 0
    
    for _ in range(num_steps):
        # 生成新动作（模拟实际使用场景）
        actions.normal_(0, Config.MAX_ACCEL)
        
        start_step = time.perf_counter()
        
        # 物理模拟计时
        physics_start = time.perf_counter()
        env._physics_step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        physics_end = time.perf_counter()
        
        # 碰撞检测计时
        collision_start = time.perf_counter()
        collision_matrix, violation_flags = env._check_violation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        collision_end = time.perf_counter()
        
        # 奖励计算计时
        reward_start = time.perf_counter()
        robot0_pos = env.positions[:, 0, :]
        dist_to_target = torch.norm(robot0_pos - env.targets, dim=1)
        speed = torch.norm(env.velocities[:, 0, :], dim=1)
        reach_target = (dist_to_target < Config.ROBOT_RADIUS) & (speed < Config.TARGET_VEL_THRESHOLD)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reward_end = time.perf_counter()
        
        total_time += time.perf_counter() - start_step
        physics_time += physics_end - physics_start
        collision_time += collision_end - collision_start
        reward_time += reward_end - reward_start
    
    # 打印结果
    print(f"\n=== Benchmark Results ({num_envs} envs) ===")
    print(f"Total steps: {num_steps}")
    print(f"Average step time: {total_time/num_steps*1000:.2f} ms")
    print(f"Steps per second: {num_steps/total_time:.2f}")
    print(f"Physics step: {physics_time/num_steps*1000:.2f} ms ({physics_time/total_time*100:.1f}%)")
    print(f"Collision detection: {collision_time/num_steps*1000:.2f} ms ({collision_time/total_time*100:.1f}%)")
    print(f"Reward calculation: {reward_time/num_steps*1000:.2f} ms ({reward_time/total_time*100:.1f}%)")
    print(f"Total env steps per second: {num_envs*num_steps/total_time:.2f}")

if __name__ == "__main__":
    # 配置测试参数
    benchmark(num_envs=4096, num_steps=100000, warmup=10)