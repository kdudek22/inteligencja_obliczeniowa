import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


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




if __name__ == '__main__':

    t = Taxi(render=True)

