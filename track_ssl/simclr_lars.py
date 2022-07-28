"""
# Another implementation of "A Simple Framework for Contrastive Learning of Visual Representatoins" SimCLR
> use LARS optimzer
"""
from functools import partial

from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch.nn import functional as F

from .ssltrainer import *
from .simclr import SimCLRParams, SimCLRTrainer, SimCLRModule


class SimCLRLarsParams(SimCLRParams):

    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim(
            'LARS',
            eta=0.001,
            use_nesterov=True,
            lr=1.0,
            weight_decay=1e-6,
        )
        self.cls_optim = self.OPTIM.create_optim(
            'SGD',
            nesterov=True,
            lr=0.01,
            momentum=0.9,
            weight_decay=0,
        )


ParamsType = SimCLRLarsParams


class SimCLRLarsTrainer(SimCLRTrainer):

    def imodels(self, params: ParamsType):
        self.model = SimCLRModule(params.model,
                                  feature_dim=params.feature_dim,
                                  hidden_size=params.hidden_feature_size,
                                  with_bn=params.with_bn,
                                  n_classes=params.n_classes)

        from models.lars import LARS
        params_models = []
        reduced_params = []

        removed_params = []

        skip_lists = ['bn', 'bias']

        for m in [self.model.backbone, self.model.head]:

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
        self.cls_optim = params.cls_optim.build(self.model.classifier.parameters())
        self.to_device()

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        xs, ys = batch['xs'], batch['ys']

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

        Lall = Lx + Lcs

        self.optim.zero_grad()
        self.cls_optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        self.cls_optim.step()
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


TrainerType = SimCLRLarsTrainer

main = partial(main, TrainerType, ParamsType)
