# import gymnasium as gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import load_model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class Taxi:
    def __init__(self, is_training=True, render=False, episodes=1000, discount_factor=0.9, learning_rate=0.06):
        self.is_training = is_training
        self.render = render
        self.episodes = episodes
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.rewards_for_episode = np.zeros(episodes)
        self.chances = np.zeros(episodes)
        self.rng = np.random.default_rng()
        self.env = gym.make("Taxi-v3", render_mode="human" if self.render else None)
        self.q = self.get_or_create_q_table()
        self.run()

    def get_or_create_q_table(self):
        if self.is_training:
            return np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            with open("q-taxi.pkl", "rb") as f:
                q = pickle.load(f)
            return q

    def run(self):
        for i in range(self.episodes):
            state, info = self.env.reset()
            terminated, truncated = False, False

            while not terminated and not truncated:
                action = self.get_next_action(i, state)

                new_state, reward, terminated, truncated, info = self.env.step(action)

                if self.is_training:
                    self.update_q_table(state, new_state, reward, action)

                state = new_state

                self.rewards_for_episode[i] = reward

        self.env.close()

        if self.is_training:
            self.save_q_table()

        self.plot_resoults()

    def get_next_action(self, i, state):
        if not self.is_training:
            return np.argmax(self.q[state, :])
        else:
            episode_to_zero_to_one = ((np.log(1 + i))/ np.log(1 + self.episodes))**2
            self.chances[i] = episode_to_zero_to_one
            if self.rng.random() < episode_to_zero_to_one:
                return np.argmax(self.q[state, :])
            else:
                return self.env.action_space.sample()

    def update_q_table(self, state, new_state, reward, action):
        self.q[state, action] = (1 - self.learning_rate) * self.q[state, action] + self.learning_rate * (
                float(reward) + self.discount_factor * np.max(self.q[new_state, :]))

    def save_q_table(self):
        with open("q-taxi.pkl", "wb") as f:
            pickle.dump(self.q, f)

    def plot_resoults(self):
        sum_rewards = np.zeros(self.episodes)
        for t in range(self.episodes):
            lower_idx = max(0, t - 100)
            upper_idx = t + 1
            diff = upper_idx - lower_idx
            if diff != 101:
                sum_rewards[t] = (np.sum(self.rewards_for_episode[lower_idx:upper_idx]) * 100 / diff)/100
            else:
                sum_rewards[t] = (np.sum(self.rewards_for_episode[max(0, t - 100):(t + 1)]))/100

        plt.plot(sum_rewards)
        plt.savefig('taxi.png')
        plt.show()

        plt.plot(self.chances)
        plt.savefig("odds.png")


class LunarLander:
    def __init__(self, is_training=True, episodes=100, training_steps=10000):
        self.is_training = is_training
        self.episodes = episodes
        self.training_steps = training_steps
        self.env = gym.make("LunarLander-v2", render_mode="human" if not self.is_training else None)
        self.model = self.create_or_load_model()
        self.train() if self.is_training else self.run()

    def create_or_load_model(self):
        if self.is_training:
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(1, self.env.observation_space.shape[0])),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.env.action_space.n)
            ])
            return model
        return load_model("lunar.h5")

    def train(self):
        agent = DQNAgent(model=self.model, memory=SequentialMemory(limit=10000, window_length=1), policy=BoltzmannQPolicy(),
                         nb_actions=self.env.action_space.n, nb_steps_warmup=10, target_model_update=0.01, gamma=1)
        agent.compile(Adam(lr=0.001), metrics=["mae"])
        history = agent.fit(self.env, nb_steps=self.training_steps, visualize=False, verbose=1)

        agent.model.save("lunar.h5")

        run_avg_reward = running_average(history.history["episode_reward"], 50)

        # with open("lunar_results.txt", "a") as myfile:
        #     myfile.write("\n")
        #     myfile.write(",".join([str(x) for x in run_avg_reward]))

        plt.plot(history.history["episode_reward"])
        plt.savefig("lunar-reward.png")
        plt.show()

        plt.plot(run_avg_reward)
        plt.savefig("lunar-running-reward.png")
        plt.show()

        plt.plot(history.history["nb_episode_steps"])
        plt.savefig("lunar-episodes.png")
        plt.show()

        self.env.close()

    def run(self):
        for i in range(self.episodes):
            state = self.env.reset()
            terminated, truncated = False, False
            while not terminated:
                action = np.argmax(self.model.predict([[[state]]]))

                new_state, reward, terminated, info = self.env.step(action)

                state = new_state
        self.env.close()


def running_average(arr, window_size):
    """Calculate the average of widnow_size behind and the current element"""
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


if __name__ == '__main__':
    lander = LunarLander(is_training=True, episodes=10, training_steps=300000)

