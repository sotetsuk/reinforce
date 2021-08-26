from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from gym.vector.vector_env import VectorEnv


class REINFORCEABC(ABC):
    n_steps: int = 0
    n_episodes: int = 0
    n_batch_updates: int = 0
    data: Dict[str, List[torch.Tensor]] = {}
    env: Optional[VectorEnv] = None
    model: Optional[nn.Module] = None
    opt: Optional[optim.Optimizer] = None

    @abstractmethod
    def train(
        self,
        env: VectorEnv,
        model: nn.Module,
        opt: optim.Optimizer,
        n_steps_lim: int,
    ):
        pass

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def act(self, observations: torch.Tensor):
        pass

    @abstractmethod
    def update_gradient(self):
        pass

    @abstractmethod
    def compute_loss(self, reduce=True):
        pass

    @abstractmethod
    def compute_return(self):
        pass

    @abstractmethod
    def push_data(self, **kwargs):
        pass
