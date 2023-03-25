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


class BaselineParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.epoch = 200
        self.method = 'baseline'
        self.train.batch_size = 512
        self.test.batch_size = 512
        self.eval.batch_size = 512
        self.apply_da = False
        self.batch_count = 0

        self.with_coarse = False  #


ParamsType = BaselineParams


class BaselineModule(nn.Module):

    def __init__(self,
                 model_name,
                 n_classes=10, c_classes=20, with_coarse=False):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        self.feature_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(self.feature_dim, n_classes)
        self.with_coarse = with_coarse
        if with_coarse:
            self.c_classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)
        output = ResnetOutput()
        if self.with_coarse:
            c_logits = self.classifier(feature_map)
            output.c_logits = c_logits

        output.feature_map = feature_map
        output.logits = logits
        return output


class BaselineTrainer(SemiTrainer):

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = BaselineModule(params.model,
                                    n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.to_device()
        self.da_prob_list = []

        if params.with_coarse:
            from datasets.const import coarse_label_map
            fc_map = coarse_label_map[params.dataset]
            self.mmap = torch.tensor(
                [fc_map[i] for i in range(params.n_classes)],
                device=self.device,
                dtype=torch.long,
            )

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        xs, ys = batch_x['xs'], batch_x['ys']

        output = self.model.forward(xs)
        x_w_logits = output.logits

        Lx = F.cross_entropy(x_w_logits, ys)

        Lcx = 0
        if params.with_coarse:
            cys = self.mmap[ys]
            c_x_w_logits = output.c_logits
            Lcx = F.cross_entropy(c_x_w_logits, cys)

        Lall = Lx + Lcx

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lx = Lx
            if params.with_coarse:
                meter.mean.Lcx = Lcx
            meter.mean.Ax = torch.eq(x_w_logits.argmax(dim=-1), ys).float().mean()

        return meter


main = partial(main, BaselineTrainer, ParamsType)

if __name__ == '__main__':
    main()
