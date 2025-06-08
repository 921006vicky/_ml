import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

total_timesteps = 0
episodes = 0

def choose_action(observation):
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    
    # 改進策略：考慮角度和角速度
    if pole_angle + 0.5 * pole_ang_vel > 0:
        return 1  # 向右推
    else:
        return 0  # 向左推

while True:
    env.render()
    action = choose_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_timesteps += 1

    if terminated or truncated:
        episodes += 1
        print(f"🚩 Episode {episodes} 結束，撐了 {total_timesteps} 步")
        total_timesteps = 0
        observation, info = env.reset()

env.close()
