"""

"""
from functools import partial

import torch

from contrib.make_optim import make_optim
from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .nltrainer import *


class BasicCEParams(NLParams):

    def __init__(self):
        super().__init__()
        self.epoch = 300 * 2
        self.train.batch_size = 64
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5  # sharpening temperature
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.02,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        self.lambda_u = 0  # weight for unsupervised loss
        self.noisy_ratio = 0.8
        self.ema = False
        self.p_threshold = 0.5  # clean probability threshold
        self.noisy_type = 'symmetric'
        self.widen_factor = 2  # 10 needs multi-gpu

    def iparams(self):
        super().iparams()
        if self.dataset == 'cifar10':
            self.warm_up = 10 * 2
        elif self.dataset == 'cifar100':
            self.warm_up = 30 * 2


ParamsType = BasicCEParams


class BasicCEModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map
        output.logits = logits
        return output


class BasicCETrainer(NLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        from copy import deepcopy
        super().imodels(params)

        lr1 = params.optim.args.lr
        lr2 = params.optim.args.lr / 10
        self.lr_sche = params.SCHE.List([
            params.SCHE.Linear(lr1, lr1, right=300),
            params.SCHE.Linear(lr2, lr2, left=300, right=params.epoch),
        ])
        self.rampup_sche = params.SCHE.Linear(start=0, end=1, left=params.warm_up, right=params.warm_up + 16)

        self.model = BasicCEModule(params.model,
                                   n_classes=params.n_classes)

        self.model2 = deepcopy(self.model)
        self.optim = make_optim(self.model, params.optim, split=params.split_params)
        self.optim2 = make_optim(self.model2, params.optim, split=params.split_params)
        self.moving_loss_dic = torch.zeros(50000, device=self.device, dtype=torch.float)
        self.all_loss = [[], []]
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        ys = batch['ys']
        if params.aug_type == 'basic':
            xs = batch['xs']
        elif params.aug_type == 'simclr':
            xs = batch['sxs0']
        elif params.aug_type == 'randaug':
            xs = batch['sxs1']
        else:
            raise NotImplementedError()

        output = self.model.forward(xs)
        Lx = F.cross_entropy(output.logits, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lx)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)

        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lx = Lx
            if params.apply_mixup:
                logits = self.model.forward(xs).logits
            else:
                logits = output.logits
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


main = partial(main, trainer_cls=BasicCETrainer, params_cls=ParamsType)
