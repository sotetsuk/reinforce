# Copyright (c) 2021 Sotetsu KOYAMADA
# https://github.com/sotetsuk/reinforce/blob/master/LICENSE

import torch


class FutureRewardMixin:
    def compute_return(self):
        R = (
            torch.stack(self.data["rewards"])
            .t()  # (n_env, max_seq_len)
            .flip(dims=(1,))
            .cumsum(dim=1)
            .flip(dims=(1,))
        )
        return R  # (n_env, max_seq_len)


class BatchAvgBaselineMixin:
    def compute_loss(self, reduce=True):
        mask = torch.stack(self.data["mask"]).t()  # (num_env, max_seq_len)
        R = self.compute_return() * mask  # (num_env, max_seq_len)
        log_p = torch.stack(self.data["log_p"]).t()  # (n_env, seq_len)
        b = self.compute_baseline(R, mask)

        # debiasing factor
        num_envs = R.size(0)
        assert num_envs > 1
        scale = num_envs / (num_envs - 1)
        loss = -scale * (R - b) * log_p * mask
        return loss.sum(dim=1).mean(dim=0) if reduce else loss

    def compute_baseline(self, R, mask):
        num_envs = R.size(0)
        R_sum = R.sum(dim=0)  # (max_seq_len)
        n_samples_per_time = mask.sum(dim=0)  # (max_seq_len)
        assert (n_samples_per_time == 0).sum() == 0
        avg = R_sum / n_samples_per_time  # (max_seq_len)
        return avg.repeat((num_envs, 1))  # (num_envs, seq_len)


class EntLossMixin:
    def compute_loss(self, reduce=True):
        loss = super().compute_loss(reduce=False)
        ent = torch.stack(self.data["entropy"]).t()  # (num_env, max_seq_len)
        mask = torch.stack(self.data["mask"]).t()  # (num_env, max_seq_len)
        assert loss.size() == ent.size() == mask.size()
        ent_loss = -ent * mask
        loss += self.ent_coef * ent_loss
        return loss.sum(dim=1).mean(dim=0) if reduce else loss
