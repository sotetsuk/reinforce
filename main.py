# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

import gym
import torch
import torch.nn as nn
import torch.optim as optim

import reinforce as rf
from reinforce.utils import evaluate


class REINFORCE(rf.FutureRewardMixin, rf.BatchAvgBaselineMixin, rf.REINFORCE):
    def __init__(self):
        super().__init__()

    def train_episode(self, env, model, opt):
        super().train_episode(env, model, opt)
        if self.n_episodes % 100 == 0:
            R = torch.stack(self.data["rewards"]).sum(dim=0).mean()
            print(
                f"step:{self.n_steps:6d}, ep:{self.n_episodes:4d}, "
                f"R:{R:.3f}"
            )


env = rf.EpisodicSyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(10)]
)
model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.Adam(model.parameters(), lr=0.01)
algo = REINFORCE()

algo.train(env, model, opt, n_steps_lim=100_000)
eval_R = evaluate(
    rf.EpisodicSyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(10)]
    ),
    model,
    deterministic=True,
)
print(f"Eval R: {eval_R:.3f}")
