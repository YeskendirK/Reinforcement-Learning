import argparse
import gym
import numpy as np
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--discount', type=float, default=0.99, metavar='G')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--num_episodes', default=200, help='number of episodes')
parser.add_argument('--win_ticks', default=195,
                    help='CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_probs = self.layers(x)
        return F.softmax(action_probs, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()  # The smallest representable positive number such that 1.0 + eps != 1.0


def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_probs = Categorical(policy(state))
    action = action_probs.sample()
    policy.saved_log_probs.append(action_probs.log_prob(action))
    return action.item()


def reinforce():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.discount * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)  # normalize returns

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    scores = deque(maxlen=100)
    for episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = choose_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        reinforce()
        scores.append(t)
        mean_score = np.mean(scores)
        if mean_score >= args.win_ticks and episode >= 100:
            print('Ran {} episodes. Solved after {} trials ✔'.format(episode, episode - 100))
            return episode - 100
        if episode % args.log_interval == 0:
            print('[Episode {}] - Mean survival time over last {} episodes was {} ticks.'.format(episode,
                                                                                                 args.log_interval,
                                                                                                 mean_score))


if __name__ == '__main__':
    main()
