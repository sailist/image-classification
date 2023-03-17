import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """adapted from https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/models.py"""

    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,  # hidden_size
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, feature):
        return self.module(feature)


class VIB(nn.Module):
    """
    https://github.com/bojone/vib/blob/master/cnn_imdb_vib.py
    """

    def __init__(self, feature_dim, lamb=0.1):
        super().__init__()
        self.lamb = lamb
        self.to_mean = nn.Linear(feature_dim, feature_dim)
        self.to_var = nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, reduction='sum'):
        z_mean, z_log_var = self.to_mean(feature), self.to_var(feature)

        if self.training:
            kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
            if reduction == 'sum':
                kl_loss = -0.5 * kl_loss.mean(0).sum()
            u = torch.rand_like(z_mean)
        else:
            kl_loss = 0
            u = 0
        feature = z_mean + torch.exp(z_log_var / 2) * u
        return (feature, kl_loss)


class MLP2(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
            nn.BatchNorm1d(output_dim) if with_bn else nn.Identity(),
        )

    def forward(self, feature):
        return self.module(feature)


class NormMLP(MLP):
    def forward(self, feature):
        return F.normalize(super().forward(feature), p=2, dim=-1)


class ResidualLinear(nn.Module):
    def __init__(self, in_feature, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feature, in_feature, bias=bias)

    def forward(self, feature):
        out = self.linear(feature)
        out = out + feature
        return out


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

    @classmethod
    def convert_split_batchnorm(cls, module, num_splits=None):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SplitBatchNorm(module.num_features, num_splits=num_splits)
            module_output.__dict__.update(module.__dict__)

        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_split_batchnorm(child, num_splits)
            )
        del module

        return module_output
