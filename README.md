![build](https://github.com/sotetsuk/reinforce/workflows/build/badge.svg)

# REINFORCE

A simple REINFORCE algorithm implementation.

## Usage

```py
import gym
from reinforce import REINFORCE, EntropyAugmentedReturn, FutureReturn, AvgBaseline, EpisodicSyncVectorEnv

model = LinearModel()
opt = SGD()

env = gym.make_env()
env = EpisodicSyncVectorEnv(env, n_env=10)

agent = REINFORCE()
agent = EntropyAugmentedReturn(agent, beta=1.0)
agent = FutureReturn(agent)
agent = AvgBaseline(agent, debiasing=True)

for i in range(100):
    agnet.train(env, model, opt, n_steps=1000)
    evaluate(env, model, deterministic=True)
```

## License

MIT License
