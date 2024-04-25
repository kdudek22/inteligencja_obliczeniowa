import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human" if False else None)

    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 0.0001
    episodes = 15000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, info = env.reset(seed=42)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, info = env.step(action)

            q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action])

            if terminated or truncated:
                new_state, info = env.reset()

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    # sum_rewards = np.zeros(episodes)
    plt.plot(sum_rewards)
    plt.savefig("test.png")

