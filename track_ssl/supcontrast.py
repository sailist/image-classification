"""
# SupContrast: Supervised Contrastive Learning
 - [paper](https://arxiv.org/abs/2004.11362)
 - [code](https://github.com/HobbitLong/SupContrast)
"""
from functools import partial

from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2, sup_contrastive_loss
from torch.nn import functional as F

from models.components import MLP
from models.module_utils import (pick_model_name,
                                 ResnetOutput, MemoryBank, LongTensorMemoryBank)
from .ssltrainer import *


class SupContrastParams(SSLParams):

    def __init__(self):
        super().__init__()
        self.method = 'supcontrast'
        self.batch_size = 1024
        self.queue_size = 8196
        self.temperature = 0.07
        self.model = 'resnet50'
        self.optim = self.OPTIM.create_optim('SGD', lr=0.05, weight_decay=1e-4, momentum=0.9)
        self.warmup_epochs = 10

        self.with_bn = False

        self.apply_mb = True  # apply memory bank (moco trick)


ParamsType = SupContrastParams


class SupContrastModule(nn.Module):

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


class SupContrastTrainer(SSLTrainer):

    def to_feature(self, xs):
        return self.model.forward(xs).feature

    def to_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = SupContrastModule(params.model,
                                       feature_dim=params.feature_dim,
                                       with_bn=params.with_bn,
                                       n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()
        self.mb_feature = self.to_device(
            MemoryBank(queue_size=params.queue_size, feature_dim=params.feature_dim)
        )
        self.mb_label = self.to_device(
            LongTensorMemoryBank(queue_size=params.queue_size)
        )

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']  # type:torch.Tensor

        sxs0, sxs1 = batch['sxs0'], batch['sxs1']

        output0 = self.model.forward(sxs0)

        if params.apply_mb:
            with torch.no_grad():
                if params.ema:
                    output1 = self.ema_model.forward(sxs1)
                else:
                    output1 = self.model.forward(sxs1)

        else:
            output1 = self.model.forward(sxs1)

        logits = output0.logits
        query = output0.feature
        key = output1.feature
        qk_graph = ys[:, None] == ys[None, :]

        qm_graph, memory = None, None
        key_neg = True
        if params.apply_mb:
            qm_graph = ys[:, None] == self.mb_label[None, :]
            memory = self.mb_feature
            key_neg = False

        # Lcs = contrastive_loss2(query=query, key=key,
        #                         memory=memory,  # moco trick
        #                         temperature=params.temperature,
        #                         norm=True,
        #                         query_neg=False, key_neg=key_neg,
        #                         qk_graph=qk_graph, qm_graph=qm_graph,
        #                         eye_one_in_qk=False)
        Lcs = sup_contrastive_loss(query, key, temperature=params.temperature, norm=True)

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
            if params.apply_mb:
                self.mb_feature.push(key)
                self.mb_label.push(ys)

            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.Lcs = Lcs
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


TrainerType = SupContrastTrainer

main = partial(main, TrainerType, ParamsType)
