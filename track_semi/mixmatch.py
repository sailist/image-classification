from functools import partial

from lumo.contrib import EMA
from lumo.contrib.torch.tensor import onehot
from torch import nn
from torch.nn import functional as F

from contrib.tensor import sharpen, label_guesses, mixmatch_up
from models.module_utils import (
    pick_model_name,
    ResnetOutput
)
from .semitrainer import *


class MixMatchParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.method = 'mixmatch'
        self.k_size = 5
        self.tempearture = 0.5
        self.apply_da = True
        self.unloader_c = 1

        self.sharpen = 0.5
        self.mixup_beta = 0.5


ParamsType = MixMatchParams


class MixMatchModule(nn.Module):

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


class MixMatchTrainer(SemiTrainer):

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = MixMatchModule(params.model,
                                    n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        xs, ys = batch_x['xs'], batch_x['ys']
        targets = onehot(ys, params.n_classes)

        uxs = [batch_un[f'xs{i}'] for i in range(params.k_size)]
        # metric_uys = batch_un['ys']

        sup_size = xs.shape[0]

        output = self.model.forward(torch.cat(uxs))
        un_logits = output.logits

        un_targets = label_guesses(*un_logits.chunk(params.k_size))
        un_targets = sharpen(un_targets, params.tempearture)

        mixed_input, mixed_target = mixmatch_up(xs, uxs, targets, un_targets)

        sup_mixed_target, unsup_mixed_target = mixed_target[:sup_size], mixed_target[sup_size:]

        mixed_output = self.model.forward(mixed_input)
        sup_mixed_logits, unsup_mixed_logits = mixed_output.logits[:sup_size], mixed_output.logits[sup_size:]

        Lx = -torch.mean(torch.sum(F.log_softmax(sup_mixed_logits, dim=1) * sup_mixed_target, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(unsup_mixed_logits, dim=1) * unsup_mixed_target, dim=1))

        Lall = Lx + Lu

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lu = Lu
            meter.mean.Lx = Lx

        return meter


main = partial(main, MixMatchTrainer, ParamsType)
