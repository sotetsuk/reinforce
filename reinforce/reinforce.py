# Copyright (c) 2021 Sotetsu KOYAMADA

# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from gym.vector.vector_env import VectorEnv
from torch.distributions import Categorical

cross_entropy = nn.CrossEntropyLoss(reduction="none")


class REINFORCE:
    def __init__(self):
        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.n_batch_updates: int = 0
        self.data: Dict[str, List[torch.Tensor]] = {}

    def train(
        self,
        env: VectorEnv,
        model: nn.Module,
        opt: optim.Optimizer,
        n_steps_lim: int = 100_000,
    ):
        while self.n_steps < n_steps_lim:
            self.n_episodes += env.num_envs
            self.train_episode(env, model, opt)

    def train_episode(
        self, env: VectorEnv, model: nn.Module, opt: optim.Optimizer
    ):
        self.data = {}
        model.train()

        observations = env.reset()  # (num_envs, obs_dim)
        dones = [False for _ in range(env.num_envs)]
        mask = torch.FloatTensor([1.0 for _ in range(env.num_envs)])
        while not all(dones):
            logits = model(
                torch.from_numpy(observations).float()
            )  # (num_envs, action_dim)
            dist = Categorical(logits=logits)
            actions = dist.sample()  # (num_envs)
            log_p = dist.log_prob(actions)  # (num_envs)
            self.n_steps += sum([not done for done in dones])
            observations, rewards, dones, info = env.step(actions.numpy())
            self.push(log_p=log_p, actions=actions, rewards=rewards, mask=mask)
            mask = 1.0 - torch.from_numpy(dones).float()

        self.update_gradient(opt)

    def update_gradient(self, opt: optim.Optimizer):
        self.n_batch_updates += 1
        opt.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        opt.step()

    def compute_loss(self):
        mask = torch.stack(self.data["mask"]).t()  # (n_env, seq_len)
        R = self.compute_return() * mask  # (n_env, seq_len)
        log_p = torch.stack(self.data["log_p"]).t()  # (n_env, seq_len)

        return -(R * log_p).sum(dim=1).mean(dim=0)

    def compute_return(self):
        seq_len = len(self.data["rewards"])
        R = (
            torch.stack(self.data["rewards"])  # (max_seq_len, num_env)
            .sum(dim=0)  # (n_env)
            .repeat((seq_len, 1))  # (max_seq_len, num_env)
            .t()  # (num_env, max_seq_len)
        )
        return R  # (n_env, max_seq_len)

    def push(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v).float()
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)


class FutureRewardMixin:
    def compute_return(self):
        R = (
            torch.stack(self.data["rewards"])
            .t()  # (n_env, max_seq_len)
            .flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))
        )
        return R  # (n_env, max_seq_len)


class BatchAvgBaselineMixin:
    def compute_loss(self):
        mask = torch.stack(self.data["mask"]).t()  # (num_env, max_seq_len)
        R = self.compute_return() * mask  # (num_env, max_seq_len)
        log_p = torch.stack(self.data["log_p"]).t()  # (n_env, seq_len)
        b = self.compute_baseline(R, mask)

        # debiasing factor
        num_envs = R.size(0)
        assert num_envs > 1
        scale = num_envs / (num_envs - 1)

        return -scale * ((R - b) * log_p * mask).sum(dim=1).mean(dim=0)

    def compute_baseline(self, R, mask):
        R = R.detach()
        mask = mask.detach()
        num_envs = R.size(0)
        R_sum = R.sum(dim=0)  # (max_seq_len)
        n_samples_per_time = mask.sum(dim=0)  # (max_seq_len)
        assert (n_samples_per_time == 0).sum() == 0
        avg = R_sum / n_samples_per_time  # (max_seq_len)
        return avg.repeat((num_envs, 1))  # (num_envs, seq_len)
