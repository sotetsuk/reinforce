from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from gym.vector.vector_env import VectorEnv
from torch.distributions import Categorical

cross_entropy = nn.CrossEntropyLoss(reduction="none")


class REINFORCE:
    def __init__(self, gamma=1.0):
        self.gamma = gamma  # TODO: use gamma
        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.n_batch_updates: int = 0

        self._rewards_seq: List[torch.Tensor] = []
        self._action_seq: List[torch.Tensor] = []
        self._logits_seq: List[torch.Tensor] = []
        self._masks_seq: List[torch.Tensor] = []

    def compute_baseline(self):
        return 0

    def compute_return(self):
        R = (
            torch.stack(self._rewards_seq)
            .t()
            .flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))
        )
        return R  # (n_env, max_seq_len)

    def compute_neg_log_prob(self):
        seq_len = len(self._logits_seq)
        logits = torch.cat(self._logits_seq)  # (num_envs * max_seq_len, num_actions)
        actions = torch.cat(self._action_seq)  # (num_envs * max_seq_len)
        neg_log_probs = cross_entropy(logits, actions)  # (num_envs * max_seq_len)
        neg_log_probs = torch.reshape(neg_log_probs, (seq_len, -1)).t()
        return neg_log_probs  # (num_envs, max_seq_len)

    def compute_loss(self, R, b, neg_log_prob, mask):
        return ((R - b) * neg_log_prob * mask).sum(dim=1).mean(dim=0)

    def push_data(self, logits, actions, rewards, dones):
        self._rewards_seq.append(torch.from_numpy(rewards).float())
        self._action_seq.append(actions)
        self._logits_seq.append(logits)
        self._masks_seq.append(1.0 - torch.from_numpy(dones).float())

    def update_gradient(self, optimizer: optim.Optimizer):
        self.n_batch_updates += 1
        optimizer.zero_grad()

        # make batch
        R = self.compute_return()  # (num_env, max_seq_len)
        neg_log_prob = self.compute_neg_log_prob()  # (num_env, max_seq_len)
        b = self.compute_baseline()
        mask = torch.stack(self._masks_seq).t()  # (num_env, max_seq_len)

        # calculate loss
        loss = self.compute_loss(R, b, neg_log_prob, mask)

        # update grad
        loss.backward()
        optimizer.step()

        self._rewards_seq = []
        self._action_seq = []
        self._logits_seq = []
        self._masks_seq = []

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
            observations = env.reset()  # (num_envs, obs_dim)
            dones = [False for _ in range(num_envs)]
            while not all(dones):
                logits = model(
                    torch.from_numpy(observations).float()
                )  # (num_envs, action_dim)
                actions = Categorical(
                    logits=logits
                ).sample()  # (num_envs, action_dim)
                n_steps += sum([not done for done in dones])
                self.n_steps += sum([not done for done in dones])
                observations, rewards, dones, info = env.step(actions.numpy())
                self.push_data(logits, actions, rewards, dones)

            self.update_gradient(optimizer)
