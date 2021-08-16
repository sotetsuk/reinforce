# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

import gym
import numpy as np

from reinforce.vector_env import EpisodicSyncVectorEnv


class ToyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.state = 0
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = 0
        done = False
        if action % 2 == self.state % 2:
            self.state += 1
            reward = 1
        if self.state == 3:
            done = True
        return self.state, reward, done, {}


def test_EpisodicSyncVectorEnv() -> None:
    env = EpisodicSyncVectorEnv([lambda: ToyEnv() for _ in range(3)])

    obs = env.reset()
    assert np.allclose(obs, [0, 0, 0])

    obs, reward, done, info = env.step([0, 0, 0])  # correct, correct, correct
    assert np.allclose(obs, [1, 1, 1])
    assert np.allclose(reward, [1, 1, 1])
    assert np.allclose(done, [False, False, False])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([1, 0, 1])  # correct, wrong, correct
    assert np.allclose(obs, [2, 1, 2])
    assert np.allclose(reward, [1, 0, 1])
    assert np.allclose(done, [False, False, False])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([0, 0, 1])  # correct, wrong, wrong
    assert np.allclose(obs, [3, 1, 2])
    assert np.allclose(reward, [1, 0, 0])
    assert np.allclose(done, [True, False, False])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([0, 0, 1])  # undef, wrong, wrong
    assert np.allclose(obs, [0, 1, 2])
    assert np.allclose(reward, [0, 0, 0])
    assert np.allclose(done, [True, False, False])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([1, 1, 0])  # undef, correct, correct
    assert np.allclose(obs, [0, 2, 3])
    assert np.allclose(reward, [0, 1, 1])
    assert np.allclose(done, [True, False, True])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([1, 0, 1])  # undef, correct, undef
    assert np.allclose(obs, [0, 3, 0])
    assert np.allclose(reward, [0, 1, 0])
    assert np.allclose(done, [True, True, True])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([0, 1, 0])  # undef, undef, undef
    assert np.allclose(obs, [0, 0, 0])
    assert np.allclose(reward, [0, 0, 0])
    assert np.allclose(done, [True, True, True])
    assert info == [{}, {}, {}]

    obs, reward, done, info = env.step([1, 0, 0])  # undef, undef, undef
    assert np.allclose(obs, [0, 0, 0])
    assert np.allclose(reward, [0, 0, 0])
    assert np.allclose(done, [True, True, True])
    assert info == [{}, {}, {}]

    obs = env.reset()  # undef, undef, undef
    assert np.allclose(obs, [0, 0, 0])

    obs, reward, done, info = env.step([1, 1, 0])  # wrong, wrong, correct
    assert np.allclose(obs, [0, 0, 1])
    assert np.allclose(reward, [0, 0, 1])
    assert np.allclose(done, [False, False, False])
    assert info == [{}, {}, {}]
