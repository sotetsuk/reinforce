import copy
from typing import List

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

        self._rewards_seq: List[torch.Tensor] = []
        self._action_seq: List[torch.Tensor] = []
        self._logits_seq: List[torch.Tensor] = []
        self._masks_seq: List[torch.Tensor] = []

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
            mask = torch.FloatTensor([1.0 for _ in range(num_envs)])
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
                self._push_data(logits, actions, rewards, mask)
                mask = 1.0 - torch.from_numpy(dones).float()

            self.update_gradient(optimizer)

    def update_gradient(self, optimizer: optim.Optimizer):
        self.n_batch_updates += 1

        optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        optimizer.step()

        self._rewards_seq = []
        self._action_seq = []
        self._logits_seq = []
        self._masks_seq = []

    def compute_loss(self):
        mask = torch.stack(self._masks_seq).t()  # (num_env, max_seq_len)
        R = self.compute_return() * mask  # (num_env, max_seq_len)
        neg_log_p = self.compute_neg_log_p() * mask  # (num_env, max_seq_len)

        return (R * neg_log_p).sum(dim=1).mean(dim=0)

    def compute_return(self):
        seq_len = len(self._rewards_seq)
        R = (
            torch.stack(self._rewards_seq)  # (max_seq_len, num_env)
            .sum(dim=0)  # (n_env)
            .repeat((seq_len, 1))  # (max_seq_len, num_env)
            .t()  # (num_env, max_seq_len)
        )
        return R  # (n_env, max_seq_len)

    def compute_neg_log_p(self):
        seq_len = len(self._logits_seq)
        logits = torch.cat(
            self._logits_seq
        )  # (num_envs * max_seq_len, num_actions)
        actions = torch.cat(self._action_seq)  # (num_envs * max_seq_len)
        neg_log_probs = cross_entropy(
            logits, actions
        )  # (num_envs * max_seq_len)
        neg_log_probs = torch.reshape(neg_log_probs, (seq_len, -1)).t()
        return neg_log_probs  # (num_envs, max_seq_len)

    def _push_data(self, logits, actions, rewards, mask):
        self._rewards_seq.append(torch.from_numpy(rewards).float())
        self._action_seq.append(actions)
        self._logits_seq.append(logits)
        self._masks_seq.append(copy.deepcopy(mask))


class FutureRewardMixin:
    def compute_return(self):
        R = (
            torch.stack(self._rewards_seq)
            .t()  # (n_env, max_seq_len)
            .flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))
        )
        return R  # (n_env, max_seq_len)


class BatchAvgBaselineMixin:
    def compute_loss(self):
        mask = torch.stack(self._masks_seq).t()  # (num_env, max_seq_len)
        R = self.compute_return() * mask  # (num_env, max_seq_len)
        neg_log_p = self.compute_neg_log_p() * mask  # (num_env, max_seq_len)
        b = self.compute_baseline(R, mask)

        # debiasing factor
        num_envs = R.size(0)
        assert num_envs > 1
        scale = num_envs / (num_envs - 1)

        return scale * ((R - b) * neg_log_p * mask).sum(dim=1).mean(dim=0)

    def compute_baseline(self, R, mask):
        R = R.detach()
        mask = mask.detach()
        num_envs = R.size(0)
        R_sum = R.sum(dim=0)  # (max_seq_len)
        n_samples_per_time = mask.sum(dim=0)  # (max_seq_len)
        assert (n_samples_per_time == 0).sum() == 0
        avg = R_sum / n_samples_per_time  # (max_seq_len)
        return avg.repeat((num_envs, 1))  # (num_envs, seq_len)
