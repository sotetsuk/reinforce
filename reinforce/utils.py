from typing import List, Optional, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from gym.vector.vector_env import VectorEnv
from torch.distributions import Categorical

import reinforce


def evaluate(
    env: Union[gym.Env, VectorEnv],
    model: nn.Module,
    deterministic: bool = False,
    num_episodes: int = 100,
    seeds: Optional[List[int]] = None,
):
    if isinstance(env, VectorEnv):
        return evaluate_vector_env(
            env, model, deterministic, num_episodes, seeds
        )
    if isinstance(env, gym.Env):
        return evaluate_env(env, model, deterministic, num_episodes, seeds)
    else:
        raise NotImplementedError


def evaluate_env(
    env: gym.Env,
    model: nn.Module,
    deterministic: bool = False,
    num_episodes: int = 100,
    seeds: Optional[List[int]] = None,
) -> float:
    model.eval()
    R_seq = []
    for i in range(num_episodes):
        if seeds is not None:
            env.seed(seeds[i])
        obs = env.reset()
        done = False
        R = 0.0
        while not done:
            logits = model(torch.FloatTensor([obs]))  # shape = (1, obs_dim)
            probs = Categorical(logits=logits)
            if deterministic:
                action = int(probs.probs.argmax(dim=-1).item())
            else:
                action = probs.sample().item()
            obs, r, done, info = env.step(action)
            R += r
        R_seq.append(R)
    return float(np.mean(R_seq))


def evaluate_vector_env(
    env: VectorEnv,
    model: nn.Module,
    deterministic: bool = False,
    num_episodes=100,
    seeds: Optional[List[int]] = None,
) -> float:
    assert isinstance(
        env,
        (reinforce.EpisodicAsyncVectorEnv, reinforce.EpisodicSyncVectorEnv),
    )
    model.eval()
    num_envs = env.num_envs
    assert num_episodes % num_envs == 0
    R_seq = []
    for i in range(num_episodes // num_envs):
        if seeds is not None:
            env.seed([i * num_envs + j for j in range(num_envs)])
        obs = env.reset()  # (num_envs, obs_size)
        done = [False for _ in range(num_envs)]
        R = np.zeros(num_envs)
        while not all(done):
            logits = model(
                torch.from_numpy(obs).float()
            )  # shape = (num_envs, obs_size)
            probs = Categorical(logits=logits)
            if deterministic:
                actions = probs.probs.argmax(dim=-1)
            else:
                actions = probs.sample()
            obs, r, done, info = env.step(actions.numpy())
            R += r  # If some episode is terminated, all r is zero afterwards.
        R_seq.append(R)

    score = np.concatenate(R_seq)
    assert score.shape[0] == num_episodes
    return float(score.mean())
