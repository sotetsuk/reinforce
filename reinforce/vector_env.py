# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

# This implementation is based on a fork from github.com/openai/gym
# Copyright (c) 2016 OpenAI (https://openai.com)
# https://github.com/openai/gym/blob/master/LICENSE.md

from copy import deepcopy

import numpy as np
from gym.vector.utils import concatenate, create_empty_array
from gym.vector.vector_env import VectorEnv


class EpisodicAsyncVectorEnv(VectorEnv):
    pass


class EpisodicSyncVectorEnv(VectorEnv):
    """Episodic version of gym.vector_env.SyncVectorEnv.
    Each episode waits until all of the episodes end.
    After the terminal state, the observation and reward will be filled with 0.
    and done will keep True value.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(
        self, env_fns, observation_space=None, action_space=None, copy=True
    ):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy

        if (observation_space is None) or (action_space is None):
            observation_space = (
                observation_space or self.envs[0].observation_space
            )
            action_space = action_space or self.envs[0].action_space
        super(EpisodicSyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_observation_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            if self._dones[i]:
                # skip env.step because i-th env is already done
                observation = create_empty_array(
                    self.single_observation_space, n=None, fn=np.zeros
                )
                self._rewards[i] = 0
                info = {}
            else:
                observation, self._rewards[i], self._dones[i], info = env.step(
                    action
                )
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError(
            "Some environments have an observation space "
            "different from `{0}`. In order to batch observations, the "
            "observation spaces from all environments must be "
            "equal.".format(self.single_observation_space)
        )
