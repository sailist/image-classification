"""
# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
 - [paper](https://arxiv.org/abs/1911.05722)
 - [code]https://github.com/facebookresearch/moco (official)

 - https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=lzFyFnhbk8hj
"""

from functools import partial

import torch
from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch.nn import functional as F
from torch import nn
from models.batch_shuffle import batch_shuffle_single_gpu, batch_unshuffle_single_gpu
from models.components import MLP, SplitBatchNorm
from models.memory_bank import MemoryBank
from models.module_utils import pick_model_name
from .ssltrainer import *


class MocoParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'moco'
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.06,
                                             weight_decay=5e-4,
                                             momentum=0.9)
        self.train.batch_size = 512
        self.queue_size = 4096  # 65535
        self.temperature = 0.1
        self.with_bn = False
        self.ema_alpha = 0.99
        self.hidden_feature_size = 512
        self.feature_dim = 128
        self.symmetric = False
        self.warmup_epochs = 0


ParamsType = MocoParams


class MocoModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=64,
                 hidden_size=2048,
                 n_classes=10,
                 with_bn=False,
                 detach_cls=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        SplitBatchNorm.convert_split_batchnorm(self.backbone, 8)

        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.head = nn.Linear(input_dim, feature_dim, bias=True)
        # self.head = MLP(input_dim,
        #                 input_dim,
        #                 output_dim=feature_dim,
        #                 with_bn=with_bn)
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


class MocoTrainer(SSLTrainer):

    def to_feature(self, xs):
        return self.model.forward(xs).feature

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = MocoModule(params.model,
                                feature_dim=params.feature_dim,
                                hidden_size=params.hidden_feature_size,
                                with_bn=params.with_bn,
                                n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.mem = MemoryBank()
        # do not need normalize because normalize is applied in contrastive_loss2 function
        self.mem.register('negative', dim=params.feature_dim, k=params.queue_size)
        self.mem['negative'] = F.normalize(self.mem['negative'], dim=-1)

        if params.ema:
            self.ema_model = EMA(self.model, alpha=params.ema_alpha)

        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        if params.ema:
            self.ema_model.step()

        ys = batch['ys']

        im_query, im_key = batch['sxs0'], batch['sxs1']

        output_query = self.model.forward(im_query)

        with torch.no_grad():
            # shuffle for making use of BN
            im_key_, idx_unshuffle = batch_shuffle_single_gpu(im_key)

            feat_key = self.ema_model.forward(im_key_).feature  # keys: NxC
            feat_key = F.normalize(feat_key, dim=1)  # already normalized

            # undo shuffle
            feat_key = batch_unshuffle_single_gpu(feat_key, idx_unshuffle)

        logits = output_query.logits
        feat_query = output_query.feature
        feat_query = F.normalize(feat_query, dim=1)

        if params.symmetric:
            Lcsa = contrastive_loss2(query=feat_query, key=feat_key,
                                     memory=self.mem['negative'],
                                     query_neg=False, key_neg=False,
                                     temperature=params.temperature,
                                     norm=False)
            Lcsb = contrastive_loss2(query=feat_key, key=feat_query,
                                     memory=self.mem['negative'],
                                     query_neg=False, key_neg=False,
                                     temperature=params.temperature,
                                     norm=False)
            Lcs = Lcsa + Lcsb
        else:

            Lcs = contrastive_loss2(query=feat_query, key=feat_key.detach(),
                                    memory=self.mem['negative'].clone().detach(),
                                    query_neg=False, key_neg=False,
                                    temperature=params.temperature,
                                    norm=False)  # norm in function)

        # memory bank
        with torch.no_grad():
            if params.symmetric:
                self.mem.push('negative', torch.cat([feat_query, feat_key], dim=0))
            else:
                self.mem.push('negative', feat_key)

        Lx = 0
        if params.train_linear:  # train disconnect classifier
            Lx = F.cross_entropy(logits, ys)

        Lall = Lx + Lcs

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)

        # metrics
        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.Lcs = Lcs
            meter.mean.Ax = torch.eq(logits.argmax(dim=-1), ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = MocoTrainer

main = partial(main, TrainerType, ParamsType)
