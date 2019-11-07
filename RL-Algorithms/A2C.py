import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='Actor-Critic')
parser.add_argument('--discount', type=float, default=0.99, metavar='G', )
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--num_episodes', default=200, help='number of episodes')
parser.add_argument('--win_ticks', default=195,
                    help='CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials')
parser.add_argument('--hidden_size', default=128)
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, env.action_space.n),
            nn.Softmax(dim=1),
        )

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        state_values = self.critic(x)
        action_probs = self.actor(x)

        return action_probs, state_values


model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()  # The smallest representable positive number such that 1.0 + eps != 1.0


def choose_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(state)

    dist = Categorical(probs)
    action = dist.sample()
    model.saved_actions.append(SavedAction(dist.log_prob(action), state_value))

    return action.item()


def A2C():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + args.discount * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value.squeeze(0), torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


def main():
    scores = deque(maxlen=100)
    for episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = choose_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        A2C()
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
