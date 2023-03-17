"""
# An implementation of "A Simple Framework for Contrastive Learning of Visual Representatoins" SimCLR
 - [paper](https://arxiv.org/pdf/2002.05709.pdf)
 - [code1](https://github.com/AidenDurrant/SimCLR-Pytorch)
 - [code2](https://github.com/leftthomas/SimCLR)

## Result
 - resnet18/cifar10/Adam
    - linear probe: 88.70%
"""
from functools import partial

import torch
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch.nn import functional as F

from models.components import MLP, MLP2
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .ssltrainer import *


class SimCLRParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'simclr'
        self.temperature = 0.5
        self.epoch = 1000
        self.train.batch_size = 512
        self.test.batch_size = 512
        self.warmup_epochs = 0
        self.hidden_feature_size = 128
        self.optim = self.OPTIM.create_optim('SGD', lr=0.6, momentum=0.9, weight_decay=1e-6)

        self.with_bn = False


ParamsType = SimCLRParams


class SimCLRModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=128,
                 hidden_size=128,
                 n_classes=10,
                 with_bn=False,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.head = MLP(input_dim,
                        hidden_size,
                        output_dim=feature_dim,
                        with_bn=with_bn)
        self.classifier = nn.Linear(input_dim, n_classes)
        self.detach_cls = detach_cls

    def forward(self, xs):
        feature_map = self.backbone(xs)
        feature = self.head(feature_map)

        if self.detach_cls:
            logits = self.classifier(feature_map.detach())
        else:
            logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map
        output.feature = feature
        output.logits = logits
        return output


class SimCLRTrainer(SSLTrainer):
    def to_feature(self, xs):
        return self.model.forward(xs).feature

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = SimCLRModule(params.model,
                                  feature_dim=params.feature_dim,
                                  hidden_size=params.hidden_feature_size,
                                  with_bn=params.with_bn,
                                  n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        ys = batch['ys']

        sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output0 = self.model.forward(sxs0)
        output1 = self.model.forward(sxs1)

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

        if params.apply_unmix:
            pass
        elif params.apply_mixco:
            pass

        Lall = Lx + Lcs

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
            meter.mean.Ax = torch.eq(logits.argmax(dim=-1), ys).float().mean()
            meter.last.lr = cur_lr
            # meter.mean.Aknn = metric_knn(query, ys, 4)

        return meter


TrainerType = SimCLRTrainer

main = partial(main, TrainerType, ParamsType)
