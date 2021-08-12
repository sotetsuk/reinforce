from typing import List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def n_steps(self):
        pass

    def n_episodes(self):
        pass

    def n_batch_updates(self):
        pass

    def train(
        self,
        env: VectorEnv,
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_steps_limit: int = 1e5,
    ):
        model.train()
        for _ in range(100):
            obs = env.reset()  # shape = (n_envs, obs_dim)
            all_done = False

            while not all_done:
                logits = model(
                    torch.from_numpy(obs).float()
                )  # shape = (n_envs, action_dim)
                a = Categorical(
                    logits=logits
                ).sample()  # shape = (n_envs, action_dim)
                obs, r, done, info = env.step(a.numpy())
                all_done = all(done)
                self.push_data(logits, a, r, done)

            self.update_gradient(optimizer)
