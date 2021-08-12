from reinforce.vector_env import EpisodicSyncVectorEnv
import gym
import numpy as np


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
        if action % 2 == self.state %2:
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
