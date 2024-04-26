import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import pickle
from CustomTetris import *
import numpy as np




class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, grid_cols=5, grid_rows=5, render_mode=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

        self.custom_tetris = CustomTetris(grid_rows, grid_cols)

        self.action_space = spaces.Discrete(len(Action))
        """For not  make it square always"""
        self.observation_space = spaces.Box(
            low=0, high=np.array([self.grid_cols]*(self.grid_rows+2)), shape=(self.grid_rows + 2,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.custom_tetris.reset(seed=seed)

        obs = self.custom_tetris.get_observations()

        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        won = self.custom_tetris.perform_action(Action(action))

        reward = 0
        terminated = self.custom_tetris.is_over

        if won == True:
            reward = 100

        elif not terminated:
            reward = 1

        obs = self.custom_tetris.get_observations()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def render(self):
        self.custom_tetris.render()


if __name__ == "__main__":
    register(id="custom-tetris-v0", entry_point="CustomEnv:TetrisEnv")

    is_training = True
    grid = 4
    env = gym.make("custom-tetris-v0", render_mode=None)
    if is_training:
        q = np.zeros((5, 5, 5, 5, 5, 5, 5, 3))
    else:
        f = open('q.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    print("checking env...")
    check_env(env)

    learning_rate_a = 0.4
    discount_factor_g = 1

    epsilon = 1
    epsilon_decay_rate = 0.0001
    episodes = 90000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # print("====== NEW EPISODE ======")
        state, info = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # action = env.action_space.sample()
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[tuple(state)])
            # print(action)

            new_state, reward, terminated, truncated, info = env.step(action)
            state_index = tuple(state)

            q[state_index][action] = q[state_index][action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[tuple(new_state)]) - q[state_index][action])

            rewards_per_episode[i] += reward

            if terminated or truncated:
                new_state, info = env.reset()

            state = new_state


        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

    env.close()

    if is_training:
        f = open('q.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])

    # sum_rewards = np.zeros(episodes)
    plt.plot(sum_rewards)
    plt.savefig("test.png")
