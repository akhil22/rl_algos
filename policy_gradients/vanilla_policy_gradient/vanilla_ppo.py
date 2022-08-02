import torch
import gym
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
eps = np.finfo(np.float32).eps.item()

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 128),
            nn.Dropout(p = 0.6),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        score = self.linear_relu_stack(x)
        probs = F.softmax(score, dim = 1)
        return probs
class Reinforce:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.policy = PolicyNetwork()
        self.max_ep_steps = 10000
        self.gamma = 1.0 
    def collect_episode(self):
        obs = self.env.reset()
        r = []
        logp = []
        for i in range(0,self.max_ep_steps):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            probs = self.policy(obs)
            m = torch.distributions.Categorical(probs = probs)
            action = m.sample()
            logp.append(m.log_prob(action))
            obs, rew, done, info = self.env.step((action.item()))
            r.append(rew)
            if done:
                break
        g = [0]
        for rew in reversed(r):
            g = [rew + self.gamma*g[0]] + g
        g = g[0:-1]
        g = torch.tensor(g)
        g = (g - g.mean()) / (g.std() + eps)
        return logp, g, sum(r)
    def show_progress(self):
        obs = self.env.reset()
        for i in range(0,1000):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            probs = self.policy(obs)
            m = torch.distributions.Categorical(probs = probs)
            action = m.sample()
            obs, rew , done ,info = self.env.step(action.item())
            self.env.render()
            #if done:
                #obs = self.env.reset()
def main():
    vanilla_pgradient = Reinforce()
    optimizer = optim.Adam(vanilla_pgradient.policy.parameters(), lr=1e-2)
    ep_rew = []
    running_reward = 10
    for i in range(0,20000):
        logp, g, r= vanilla_pgradient.collect_episode()
        loss = 0
        for log_p, R in zip(logp, g):
            loss = loss - log_p*R
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_reward = 0.05 * r + (1 - 0.05) * running_reward
        if(running_reward > vanilla_pgradient.env.spec.reward_threshold):
            print("solved")
            break
        if(i % 10 == 0):
            print(f'episode: {i}\t ep_reward: {r:.2f}\t average_reward: {running_reward:.2f}')
        #print("logp\n", logp)
        #print("g\n", g)
        ep_rew.append(r)
        del logp[:]
    vanilla_pgradient.show_progress()
    plt.plot(ep_rew)
    plt.show()
    plt.close()
    print(vanilla_pgradient.env.spec.reward_threshold)
if __name__ == '__main__':
    main()