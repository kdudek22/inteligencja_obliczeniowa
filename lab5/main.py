from datetime import datetime

import gymnasium as gym
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

env = gym.make("Humanoid-v4")

discount_factor = 0.99
learning_rate = 3e-3
buffer_size = 1_000_000
i = 4

model = SAC("MlpPolicy", env, verbose=0, gamma=discount_factor, learning_rate=learning_rate, buffer_size=buffer_size, tensorboard_log=f"logs/{i}")

max_steps = 1000
mean_rewards, std_rewards = [], []

start = datetime.now()

for i in range(50):
    episode_start = datetime.now()
    model.learn(total_timesteps=max_steps, reset_num_timesteps=False)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    print("LEN:" + str(datetime.now() - episode_start))

print(mean_rewards)
print(std_rewards)
print(datetime.now() - start)

discount_factor_str = str(discount_factor).replace(".", "_")
learning_rate_str = str(learning_rate).replace(".", "_")
buffer_size_str = str(buffer_size).replace(".", "_")


"""Discount_factor, learning_rate, buffer_size"""
with open(f"SAC__{discount_factor_str}__{learning_rate_str}__{buffer_size_str}.txt", "w") as f:
    f.write(",".join([str(x) for x in mean_rewards]))

# plt.plot([i for i in range(len(mean_rewards))], mean_rewards)
# plt.savefig("test.png")

"""
1. DC 0.99 lr 0.0003 
2. DC 0.07 lr 0.0003
3. DC 07 lr 0.003
4. DC 0.99 lr 0.003
"""