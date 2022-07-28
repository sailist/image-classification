"""
# EQUIVARIANT CONTRASTIVE LEARNING (E-SSL)
 - [paper](https://arxiv.org/pdf/2111.00899.pdf)

> Currently cannot reproduce the result.

## Result:
 - resnet18/cifar10
    - linear probe: 42.37%

"""
from functools import partial

import torch
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch import nn
from torch.nn import functional as F

from models.components import MLP, MLP2
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .ssltrainer import *


class ESSLParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'essl'
        self.epoch = 800
        self.temperature = 0.5
        self.optim = self.OPTIM.create_optim('SGD', lr=0.03, weight_decay=5e-4)

        self.feature_dim = 2048
        self.with_bn = False
        self.w_dce = 0.4  # lambda, weight for E-SSL loss


ParamsType = ESSLParams


class ESSLResnetOutput(ResnetOutput):
    direct_logits: torch.Tensor = None


class ESSLModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=64,
                 n_classes=10,
                 with_bn=False,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.head = MLP(input_dim,
                        feature_dim,
                        output_dim=feature_dim,
                        with_bn=with_bn)
        self.classifier = nn.Linear(input_dim, n_classes)
        self.direct_classifier = nn.Linear(input_dim, 4)
        self.detach_cls = detach_cls

    def forward(self, xs, output_direct=False):
        feature_map = self.backbone(xs)
        feature = self.head(feature_map)

        if self.detach_cls:
            logits = self.classifier(feature_map.detach())
        else:
            logits = self.classifier(feature_map)

        direct_logits = None
        if output_direct:
            direct_logits = self.direct_classifier(feature_map)

        output = ESSLResnetOutput()
        output.feature_map = feature_map
        output.feature = feature
        output.logits = logits
        output.direct_logits = direct_logits
        return output


class ESSLTrainer(SSLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = ESSLModule(params.model,
                                feature_dim=params.feature_dim,
                                with_bn=params.with_bn,
                                n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs0, xs1 = batch['sxs0'], batch['sxs1']
        ys = batch['ys']

        dir_xs = []
        dir_ys = []
        for i, direct in enumerate('udlr'):
            dir_xs.append(batch[f'xs{direct}'])
            dir_ys.append(torch.full((len(dir_xs[-1]),), i, device=xs0.device, dtype=torch.long))
        dir_xs, dir_ys = torch.cat(dir_xs), torch.cat(dir_ys)

        output0 = self.model.forward(xs0)
        output1 = self.model.forward(xs1)

        output_dir = self.model.forward(dir_xs, output_direct=True)

        Ldce = F.cross_entropy(output_dir.direct_logits, dir_ys) / 4

        logits = output0.logits
        query = output0.feature
        key = output1.feature

        Lcs = contrastive_loss2(query=query, key=key,
                                temperature=params.temperature,
                                query_neg=True, key_neg=True,
                                norm=True,
                                eye_one_in_qk=False)

        Lx = 0
        if params.train_linear:
            Lx = F.cross_entropy(logits, ys)

        Lall = Lx + Lcs + Ldce

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.Lcs = Lcs
            meter.mean.Ldce = Ldce
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.mean.Ad = (output_dir.direct_logits.argmax(dim=-1) == dir_ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = ESSLTrainer

main = partial(main, TrainerType, ParamsType)
