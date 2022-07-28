"""
linear probe with LARS
"""
import os.path

import torch
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch import nn
from torch.nn import functional as F

from models.components import MLP, MLP2
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .ssltrainer import *


class GeneralParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim(
            'LARS',
            eta=0.001,
            use_nesterov=True,
            lr=1.0,
            weight_decay=1e-6,
        )
        self.warmup_epochs = 0
        self.epoch = 100
        self.ema = False
        self.batch_size = 256
        self.backbone_path = None


ParamsType = GeneralParams


class GeneralModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=64,
                 n_classes=10,
                 with_bn=False,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.head = MLP2(input_dim,
                         input_dim,
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


class GeneralTrainer(SSLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = GeneralModule(params.model,
                                   feature_dim=params.feature_dim,
                                   with_bn=params.with_bn,
                                   n_classes=params.n_classes)

        if os.path.exists(params.backbone_path):
            self.model.backbone.load_state_dict(torch.load(params.backbone_path, 'cpu'))
            self.logger.info('load backbone parameters')

        from models.lars import LARS
        params_models = []
        reduced_params = []

        removed_params = []

        skip_lists = ['bn', 'bias']

        for m in [self.model.classifier]:
            m_skip = []
            m_noskip = []

            params_models += list(m.parameters())

            for name, param in m.named_parameters():
                if (any(skip_name in name for skip_name in skip_lists)):
                    m_skip.append(param)
                else:
                    m_noskip.append(param)
            reduced_params += list(m_noskip)
            removed_params += list(m_skip)

        params.optim.len_reduced = len(reduced_params)
        self.optim = params.optim.build(reduced_params + removed_params, LARS)

        self.optim = params.optim.build(self.model.classifier.parameters())
        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']

        # sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output0 = self.model.forward(xs)
        logits = output0.logits

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
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = GeneralTrainer
