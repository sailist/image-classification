"""
refer to
 - https://github.com/kekmodel/FixMatch-pytorch
"""
from functools import partial

from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from track_semi.semitrainer import *


class FixMatchParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.epoch = 1024
        self.method = 'fixmatch'
        self.apply_da = False


ParamsType = FixMatchParams


class FixMatchModule(nn.Module):

    def __init__(self,
                 model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        self.feature_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map
        output.logits = logits
        return output


class FixMatchTrainer(SemiTrainer):

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = FixMatchModule(params.model,
                                    n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.da_prob_list = []
        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        xs, ys = batch_x['xs'], batch_x['ys']

        idu = batch_un['id']
        uxs, uxs_s0 = batch_un['xs'], batch_un['sxs']
        metric_uys = batch_un['ys']

        sup_size = xs.shape[0]

        axs = torch.cat([xs, uxs, uxs_s0])
        output = self.model.forward(axs)
        (x_w_logits), (un_w_logits, un_s0_logits) = (
            output.logits[:sup_size],
            output.logits[sup_size:].chunk(2)
        )

        def data_align():
            probs = un_w_probs
            self.da_prob_list.append(probs.mean(0))
            if len(self.da_prob_list) > 32:
                self.da_prob_list.pop(0)

            prob_avg = torch.stack(self.da_prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            return probs

        un_w_probs = torch.softmax(un_w_logits, dim=1)
        if params.apply_da:
            un_w_probs = data_align()

        scores, un_w_pys = torch.max(un_w_probs, dim=1)
        p_mask = scores.ge(params.pseudo_thresh)

        if params.record_predict and 'stl' not in params.dataset:
            with torch.no_grad():
                self.tensors.scatter('pscore', scores, idu)
                self.tensors.scatter('pys', un_w_pys, idu)

        Lu = F.cross_entropy(un_s0_logits, un_w_pys, reduction='none') * p_mask.float()
        Lu = Lu.mean()
        Lx = F.cross_entropy(x_w_logits, ys)

        Lall = Lx + Lu

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            meter.mean.Lu = Lu
            meter.mean.Ax = torch.eq(x_w_logits.argmax(dim=-1), ys).float().mean()
            meter.mean.Au = torch.eq(un_w_logits.argmax(dim=-1), metric_uys).float().mean()
            if p_mask.any():
                meter.mean.Aum = torch.eq(un_w_logits.argmax(dim=-1), metric_uys)[p_mask].float().mean()
                meter.mean.um = p_mask.float().mean()

        return meter


main = partial(main, FixMatchTrainer, ParamsType)

if __name__ == '__main__':
    main()
