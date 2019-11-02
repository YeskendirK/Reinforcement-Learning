'''
Balanced Cart-Pole using  Sarsa(lambda) RL algorithm
'''

import gym
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.style.use('ggplot')


class Sarsa_lambda():
    def __init__(self, episodes=10000, min_lr=0.01, min_epsilon=0.4, discount=1, lmbd=0.95,
                 max_env_steps=100, quiet=False, monitor=False, ada_divisor=25):
        self.episodes = episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.lmbd = lmbd
        self.quiet = quiet
        self.ada_divisor = ada_divisor

        self.env = gym.make('FrozenLake-v0')
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, 'tmp/frozenlake-1', force=True)

        self.sarsa_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.E = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (random.uniform(0, 1) <= epsilon) else np.argmax(
            self.sarsa_table[state, :])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_lr(self, t):
        return self.min_lr

    def run(self):
        print('Sarsa(lambda)')
        stats = [0] * self.episodes

        for episode in range(self.episodes):
            self.E = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            lr = self.get_lr(episode)
            epsilon = self.get_epsilon(episode)
            done = False
            time_step = 0

            current_state = self.env.reset()
            action = self.choose_action(current_state, epsilon)

            while not done:
                new_state, reward, done, info = self.env.step(action)

                new_action = self.choose_action(new_state, epsilon)
                delta = reward + self.discount * self.sarsa_table[new_state, new_action] - \
                        self.sarsa_table[current_state, action]
                self.E[current_state, action] += 1
                for s in range(self.env.observation_space.n):
                    for a in range(self.env.action_space.n):
                        self.sarsa_table[s, a] += lr * delta * self.E[s, a]
                        if s == current_state:
                            self.E[s,a] = 1
                        else:
                            self.E[s, a] = self.discount * self.lmbd * self.E[s, a]

                current_state, action = new_state, new_action
                stats[episode] = time_step
                time_step += 1
                if done:
                    break
            if (episode + 1) % 1000 == 0:
                print("Episode {}/{}.".format(episode + 1, self.episodes))
        return self.sarsa_table, stats


if __name__ == "__main__":
    solver = Sarsa_lambda()
    table, stats = solver.run()
    fig1 = plt.figure(figsize=(10, 10))
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.plot(stats)
    plt.show(fig1)

    # Test
    env = gym.make('FrozenLake-v0')
    count_success = 0
    eps = 10
    for episode in range(eps):
        state = env.reset()
        step = 0
        done = False
        print('***************************************')
        print('EPISODE', episode)

        for step in range(100):
            action = np.argmax(table[state, :])
            new_state, reward, done, info = env.step(action)
            if reward > 0:
                count_success += 1
            if done:
                env.render()

                # The number of step it took.
                print("Number of steps", step)
                break
            state = new_state
        env.close()
    print('Success Rate: {} / {}'.format(count_success, eps))
