'''
Balanced Cart-Pole using naive Q-learning RL algorithm
With hyperparameters: min_epsilon=0.1; discount=1.0, min_lr=0.1, buckets=[1,1,6,12] balanced in 220 episodes (120 trials)
'''

import gym
import numpy as np
import math
from collections import deque


class CartPole():
    def __init__(self, buckets=(1, 1, 6, 12), episodes=1000, win_ticks=195,
                 min_lr=0.1, min_epsilon=0.1, discount=1.0, max_env_steps=None, quiet=False, monitor=False,
                 ada_divisor=25):
        self.buckets = buckets
        self.episodes = episodes
        self.win_ticks = win_ticks
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.quiet = quiet
        self.ada_divisor = ada_divisor

        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True)

        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def state_to_buckets(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_table[state])

    def update_q(self, old_state, action, reward, new_state, lr):
        self.Q_table[old_state][action] += \
            lr * (reward + self.discount * np.max(self.Q_table[new_state]) - self.Q_table[old_state][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_lr(self, t):
        return max(self.min_lr, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def run(self):
        print('Q-learning')
        scores = deque(maxlen=100)

        for episode in range(self.episodes):
            current_state = self.state_to_buckets(self.env.reset())

            lr = self.get_lr(episode)
            epsilon = self.get_epsilon(episode)
            done = False
            time_step = 0

            while not done:
                self.env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.state_to_buckets(obs)
                self.update_q(current_state, action, reward, new_state, lr)
                current_state = new_state
                time_step += 1

            scores.append(time_step)
            mean_score = np.mean(scores)
            if mean_score >= self.win_ticks and episode >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(episode, episode - 100))
                return episode - 100
            if episode % 10 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 10 episodes was {} ticks.'.format(episode, mean_score))

        if not self.quiet: print('Did not solve after {} episodes'.format(episode))
        return episode


if __name__ == "__main__":
    solver = CartPole()
    solver.run()
