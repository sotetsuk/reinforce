import gym
import torch.nn as nn
import torch.optim as optim

import reinforce
import reinforce.utils

if __name__ == "__main__":
    n_envs = 10
    env = reinforce.EpisodicSyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(n_envs)]
    )

    model = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
    opt = optim.Adam(model.parameters(), lr=0.002)

    rf = reinforce.REINFORCE()
    n_train = 0
    while rf.n_steps < 100_000:
        rf.train(env, model, opt, n_steps_lim=n_train * 10_000)
        score = reinforce.utils.evaluate(
            reinforce.EpisodicSyncVectorEnv(
                [lambda: gym.make("CartPole-v1") for _ in range(10)]
            ),
            model,
            deterministic=False,
            num_episodes=100,
            seeds=list(range(100)),
        )
        print(
            f"n_steps={rf.n_steps:8d}\tn_episodes={rf.n_episodes:8d}\tn_batch_updates={rf.n_batch_updates:8d}\tscore={score}"
        )
        n_train += 1

    env.close()
