import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle



def run(episodes, is_training=True, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open("taxi.pkl", 'rb')
        q = pickle.load(f)
        f.close()

    discount_factor_g = 0.5
    epsilon = 1
    epsilon_decay_rate = 0.0001  # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()
    learning_rate_a = 0.4
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        reward = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (float(reward) + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0005

        rewards_per_episode[i] = reward

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        lower_idx = max(0, t-100)
        upper_idx = t+1
        diff = upper_idx - lower_idx
        if diff != 101:
            sum_rewards[t] = np.sum(rewards_per_episode[lower_idx:upper_idx]) * 100/diff
        else:
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if is_training:
        f = open("taxi.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(5000, is_training=True, render=True)