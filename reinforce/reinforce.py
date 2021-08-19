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
        self.env = None
        self.model = None
        self.opt = None

    def train(
        self,
        env: VectorEnv,
        model: nn.Module,
        opt: optim.Optimizer,
        n_steps_lim: int = 100_000,
    ):
        self.env, self.model, self.opt = env, model, opt
        while self.n_steps < n_steps_lim:
            self.train_episode()
        self.env, self.model, self.opt = None, None, None

    def train_episode(self):
        self.n_episodes += self.env.num_envs
        self.data = {}
        self.model.train()

        observations = self.env.reset()  # (num_envs, obs_dim)
        dones = [False for _ in range(self.env.num_envs)]
        mask = torch.FloatTensor([1.0 for _ in range(self.env.num_envs)])
        while not all(dones):
            logits = self.model(
                torch.from_numpy(observations).float()
            )  # (num_envs, action_dim)
            dist = Categorical(logits=logits)
            actions = dist.sample()  # (num_envs)
            log_p = dist.log_prob(actions)  # (num_envs)
            self.n_steps += sum([not done for done in dones])
            observations, rewards, dones, info = self.env.step(actions.numpy())
            self.push(log_p=log_p, actions=actions, rewards=rewards, mask=mask)
            mask = 1.0 - torch.from_numpy(dones).float()

        self.update_gradient(self.opt)

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
