"""
# Bootstrap your own latent: A new approach to self-supervised Learning
 - [paper](https://arxiv.org/abs/2006.07733)
 - [code](https://github.com/lucidrains/byol-pytorch)


"""
from functools import partial

import torch

from lumo import MetricType
from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F

from models.components import MLP
from models.module_utils import (pick_model_name,
                                 ResnetOutput,
                                 MemoryBank)
from .ssltrainer import *


class BYOLParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'byol'
        self.epoch = 800
        self.batch_size = 512

        self.optim = self.OPTIM.create_optim('SGD', lr=0.05, weight_decay=5e-4, momentum=0.9)
        self.warmup_epochs = 0
        self.hidden_feature_size = 512
        self.with_bn = True
        self.apply_simsiam = False


ParamsType = BYOLParams


class BYOLModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.feature_dim = input_dim
        self.predictor = nn.Sequential(
            MLP(input_dim,
                input_dim // 4,
                output_dim=input_dim,
                with_bn=True,
                with_leakyrelu=False  # a small change
                ),
            nn.BatchNorm1d(input_dim, affine=False)
        )
        self.classifier = nn.Linear(input_dim, n_classes)
        self.detach_cls = detach_cls

    def forward(self, xs):
        feature_map = self.backbone(xs)
        feature = self.predictor(feature_map)

        if self.detach_cls:
            logits = self.classifier(feature_map.detach())
        else:
            logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map  # z, target
        output.feature = feature  # p, predict
        output.logits = logits
        return output


def loss_fn(p, z):
    return -F.cosine_similarity(p, z, dim=-1).mean()


class BYOLTrainer(SSLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def to_resnet_output(self, xs) -> ResnetOutput:
        if self.params.ema:
            return self.ema_model.forward(xs)
        return self.model.forward(xs)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = BYOLModule(params.model,
                                n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()
        self.mb_feature = self.to_device(
            MemoryBank(queue_size=65535, feature_dim=params.feature_dim)
        )

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        xs, ys = batch['xs'], batch['ys']

        sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output_0 = self.model.forward(sxs0)
        output_1 = self.model.forward(sxs1)

        if params.apply_simsiam:
            target_0 = output_0.feature_map.detach()
            target_1 = output_1.feature_map.detach()
        else:
            with torch.no_grad():  # detach
                target_0 = self.ema_model.forward(sxs0).feature_map
                target_1 = self.ema_model.forward(sxs1).feature_map

        La = loss_fn(output_0.feature, target_1)
        Lb = loss_fn(output_1.feature, target_0)

        logits = output_0.logits

        Lx = 0
        if params.train_linear:
            Lx = F.cross_entropy(logits, ys)

        Lall = Lx + (La + Lb) / 2

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.La = La.mean()
            meter.mean.Lb = Lb.mean()
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = BYOLTrainer

main = partial(main, TrainerType, ParamsType)
