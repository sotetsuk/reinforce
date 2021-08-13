import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import reinforce


def evaluate(
    env: gym.Env,
    model: nn.Module,
    deterministic: bool = False,
    num_episodes=100,
) -> float:
    model.eval()
    R_seq = []
    for n in range(num_episodes):
        obs = env.reset()
        done = False
        R = 0.0
        while not done:
            logits = model(torch.FloatTensor([obs]))  # shape = (1, obs_dim)
            probs = Categorical(logits=logits)
            if deterministic:
                raise NotImplementedError
            action = probs.sample().item()
            obs, r, done, info = env.step(action)
            R += r
        R_seq.append(R)
    return float(torch.FloatTensor(R_seq).mean().item())


if __name__ == "__main__":
    n_envs = 10
    env = reinforce.EpisodicSyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(n_envs)]
    )

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.data = []

            self.fc1 = nn.Linear(4, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    rf = reinforce.REINFORCE()
    while rf.n_steps < 100_000:
        rf.train(env, model, optimizer, n_steps_limit=10_000)
        score = evaluate(gym.make("CartPole-v1"), model, num_episodes=1)
        print(
            f"n_steps={rf.n_steps:8d}\tn_episodes={rf.n_episodes:8d}\tn_batch_updates={rf.n_batch_updates:8d}\tscore={score}"
        )

    env.close()
