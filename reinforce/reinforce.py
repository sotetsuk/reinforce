from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from gym.vector.vector_env import VectorEnv
from torch.distributions import Categorical

cross_entropy = nn.CrossEntropyLoss(reduction="none")


class REINFORCE:
    def __init__(self):
        self.gamma = 1.0
        self.rewards: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.n_batch_updates: int = 0

    def compute_baseline(self):
        return 0

    def compute_return(self):
        R = (
            torch.stack(self.rewards)
            .t()
            .flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))
        )
        return R  # (n_env, max_seq_len)

    def compute_neg_log_prob(self):
        neg_log_probs = []
        for logit, a in zip(self.logits, self.actions):
            neg_log_probs.append(cross_entropy(logit, a))
        return torch.stack(neg_log_probs).t()  # (n_env, max_seq_len)

    def compute_loss(self, R, b, neg_log_prob, mask):
        return ((R - b) * neg_log_prob * mask).sum(dim=1).mean(dim=0)

    def push_data(self, logits, actions, rewards, dones):
        self.rewards.append(torch.from_numpy(rewards).float())
        self.actions.append(actions)
        self.logits.append(logits)
        self.masks.append(1.0 - torch.from_numpy(dones).float())

    def update_gradient(self, optimizer: optim.Optimizer):
        self.n_batch_updates += 1
        optimizer.zero_grad()

        # make batch
        R = self.compute_return()
        neg_log_prob = self.compute_neg_log_prob()
        b = self.compute_baseline()
        mask = torch.stack(self.masks).t()  # (n_env, max_seq_len)

        # calculate loss
        loss = self.compute_loss(R, b, neg_log_prob, mask)

        # update grad
        loss.backward()
        optimizer.step()

        self.rewards = []
        self.actions = []
        self.logits = []
        self.masks = []

    def train(
        self,
        env: VectorEnv,
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_steps_limit: int = 100_000,
    ):
        model.train()
        num_envs = env.num_envs
        n_steps = 0
        while n_steps < n_steps_limit:
            self.n_episodes += num_envs
            obs = env.reset()  # shape = (num_envs, obs_dim)
            dones = [False for _ in range(num_envs)]
            while not all(dones):
                logits = model(
                    torch.from_numpy(obs).float()
                )  # shape = (num_envs, action_dim)
                a = Categorical(
                    logits=logits
                ).sample()  # shape = (num_envs, action_dim)
                n_steps += sum([not done for done in dones])
                self.n_steps += sum([not done for done in dones])
                obs, r, dones, info = env.step(a.numpy())
                self.push_data(logits, a, r, dones)

            self.update_gradient(optimizer)
