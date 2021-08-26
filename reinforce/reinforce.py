# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from gym.vector.vector_env import VectorEnv
from torch.distributions import Categorical

import reinforce as rf


class REINFORCE(rf.REINFORCEABC):
    def __init__(self):
        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.n_batch_updates: int = 0
        self.data: Dict[str, List[torch.Tensor]] = {}
        self.env: Optional[VectorEnv] = None
        self.model: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None

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
        assert self.env is not None and self.model is not None
        self.n_episodes += self.env.num_envs
        self.data = {}
        self.model.train()

        observations = self.env.reset()  # (num_envs, obs_dim)
        dones = [False for _ in range(self.env.num_envs)]
        mask = torch.FloatTensor([1.0 for _ in range(self.env.num_envs)])
        while not all(dones):
            self.n_steps += sum([not done for done in dones])
            actions = self.act(torch.from_numpy(observations).float())
            observations, rewards, dones, info = self.env.step(actions.numpy())
            self.push_data(rewards=rewards, mask=mask)
            mask = 1.0 - torch.from_numpy(dones).float()
        self.update_gradient()

    def act(self, observations: torch.Tensor):
        assert self.model is not None
        logits = self.model(observations)  # (num_envs, action_dim)
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (num_envs)
        log_prob = dist.log_prob(actions)  # (num_envs)
        entropy = dist.entropy()  # (num_envs)
        self.push_data(log_prob=log_prob, actions=actions, entropy=entropy)
        return actions

    def update_gradient(self):
        assert self.opt is not None
        self.n_batch_updates += 1
        self.opt.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.opt.step()

    def compute_loss(self, reduce=True):
        mask = torch.stack(self.data["mask"]).t()  # (n_env, seq_len)
        R = self.compute_return() * mask  # (n_env, seq_len)
        log_prob = torch.stack(self.data["log_prob"]).t()  # (n_env, seq_len)
        loss = -R * log_prob
        return loss.sum(dim=1).mean(dim=0) if reduce else loss

    def compute_return(self):
        seq_len = len(self.data["rewards"])
        R = (
            torch.stack(self.data["rewards"])  # (max_seq_len, num_env)
            .sum(dim=0)  # (n_env)
            .repeat((seq_len, 1))  # (max_seq_len, num_env)
            .t()  # (num_env, max_seq_len)
        )
        return R  # (n_env, max_seq_len)
