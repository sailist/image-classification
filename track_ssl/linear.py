"""
linear probe
"""
import os.path
from functools import partial

import torch
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch import nn
from torch.nn import functional as F

from contrib.load_ssl_model import SSLLoadModel
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .ssltrainer import *


class LinearEvalParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim('SGD', lr=0.2, weight_decay=0., momentum=0.9, nesterov=True)
        self.warmup_epochs = 0
        self.epoch = 100
        self.ema = False
        self.train.batch_size = 512
        self.test.batch_size = 512

        self.pretrain = True
        self.pretrain_path = None
        self.method = 'linear_eval'


ParamsType = LinearEvalParams


class LinearEvalModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=128,
                 n_classes=10,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(input_dim, n_classes)
        self.detach_cls = detach_cls

    def forward(self, xs):
        feature_map = self.backbone(xs)

        if self.detach_cls:
            feature_map = feature_map.detach()

        logits = self.classifier(feature_map)
        output = ResnetOutput()
        output.feature_map = feature_map
        output.logits = logits
        return output


class LinearEvalTrainer(SSLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        SSLLoadModel().hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = LinearEvalModule(params.model,
                                      feature_dim=params.feature_dim,
                                      n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.classifier.parameters())
        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']

        # sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output = self.model.forward(xs)
        logits = output.logits

        Lx = F.cross_entropy(logits, ys)

        Lall = Lx

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.Ax = torch.eq(logits.argmax(dim=-1), ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = LinearEvalTrainer

main = partial(main, TrainerType, ParamsType)
