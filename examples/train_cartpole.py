# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE
from typing import Dict, List

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

import reinforce as rf
from reinforce.utils import evaluate


class REINFORCE1(rf.REINFORCE):
    def __init__(self):
        super().__init__()


class REINFORCE2(rf.FutureRewardMixin, rf.REINFORCE):
    def __init__(self):
        super().__init__()


class REINFORCE3(rf.BatchAvgBaselineMixin, rf.REINFORCE):
    def __init__(self):
        super().__init__()


class REINFORCE4(rf.FutureRewardMixin, rf.BatchAvgBaselineMixin, rf.REINFORCE):
    def __init__(self):
        super().__init__()


def make_env():
    return rf.EpisodicSyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(10)]
    )


def make_algo(algo_num):
    if algo_num == 1:
        return REINFORCE1()
    elif algo_num == 2:
        return REINFORCE2()
    elif algo_num == 3:
        return REINFORCE3()
    elif algo_num == 4:
        return REINFORCE4()
    else:
        raise NotImplementedError


for lr in [0.01, 0.03, 0.1]:
    results: Dict[str, List[List[float]]] = {}
    for algo_num in range(1, 5):
        print("====================================")
        print(f"algo{algo_num}, lr={lr}")
        print("====================================")
        for n in range(10):
            algo = make_algo(algo_num)
            env = make_env()
            model = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)
            )
            opt = optim.Adam(model.parameters(), lr=0.03)
            n_train = 0
            score_seq = []
            while algo.n_steps < 10 ** 5:
                algo.train(
                    env, model, opt, n_steps_lim=(n_train + 1) * (10 ** 4)
                )
                score = evaluate(
                    make_env(),
                    model,
                    deterministic=False,
                    num_episodes=100,
                )
                score_seq.append(score)
                n_train += 1

            print(score_seq, flush=True)
            if f"algo{algo_num}" not in results:
                results[f"algo{algo_num}"] = []
            results[f"algo{algo_num}"].append(score_seq)

    plt.figure()
    for algo_num in range(1, 5):
        plt.plot(
            np.arange(10 ** 4, 10 ** 5 + 1, 10 ** 4),
            np.vstack([np.array(x) for x in results[f"algo{algo_num}"]]).mean(
                axis=0
            ),
            label=f"algo_{algo_num}",
        )
    plt.legend()
    plt.savefig(f"cartpole_lr={lr}.png")
