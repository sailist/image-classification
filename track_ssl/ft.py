"""
# An implementation of "About Improving Contrastive Learning by Visualizing Feature Transformation" FT
 - [paper](https://arxiv.org/abs/2108.02982)
 - [code1](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)

"""
from functools import partial
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch import nn
from torch.nn import functional as F

from models.components import MLP
from models.module_utils import (pick_model_name,
                                 ResnetOutput,
                                 MemoryBank)
from .ssltrainer import *


class FTParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'ft'
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.03,
                                             weight_decay=1e-4,
                                             momentum=0.9)
        self.queue_size = 16392  # 65535
        self.temperature = 0.07
        self.hidden_feature_size = 4096
        self.with_bn = True
        self.alpha = 0.9
        self.contrast_th = 0.8
        self.ema = True


ParamsType = FTParams


class FTModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=64,
                 hidden_size=2048,
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


class FTTrainer(SSLTrainer):

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = FTModule(params.model,
                                feature_dim=params.feature_dim,
                                hidden_size=params.hidden_feature_size,
                                with_bn=params.with_bn,
                                n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()
        self.mb_feature = self.to_device(
            MemoryBank(queue_size=params.queue_size, feature_dim=params.feature_dim)
        )

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']

        sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output0 = self.model.forward(sxs0)
        with torch.no_grad():
            output1 = self.ema_model.forward(sxs1)
            key = output1.feature
            self.mb_feature.push(key)

        logits = output0.logits
        query = output0.feature

        Lcs = contrastive_loss2(query=query, key=key,
                                memory=self.mb_feature,
                                query_neg=False, key_neg=False,
                                temperature=params.temperature,
                                norm=True,
                                eye_one_in_qk=False)
        Lx = 0
        if params.train_linear:
            Lx = F.cross_entropy(logits, ys)

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
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = FTTrainer

main = partial(main, TrainerType, ParamsType)
