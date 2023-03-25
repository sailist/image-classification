"""
Treat all noisy data as clean data, can be seen as the lower bound of all noisy label learning methods.
"""
from functools import partial

from contrib.make_optim import make_optim
from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from track_nl.nltrainer import *


class CrossEntropyParams(NLParams):

    def __init__(self):
        super().__init__()
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.02,
                                             momentum=0.9,
                                             nesterov=True,
                                             weight_decay=5e-4)
        self.epoch = 400
        self.train.batch_size = 128
        self.noisy_ratio = 0.8
        self.ema = True
        self.ema_alpha = 0.999


ParamsType = CrossEntropyParams


class CrossEntropyModule(nn.Module):

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


class CrossEntropyTrainer(NLTrainer):

    def inference(self, batch):
        pass

    def predict(self, batch):
        pass

    def to_logits(self, xs):
        if self.params.get('ema', False):
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)

        self.lr_sche = params.SCHE.Cos(start=params.optim.lr,
                                       end=0.0001,
                                       right=400)

        self.model = CrossEntropyModule(params.model,
                                        n_classes=params.n_classes)

        # self.model2 = deepcopy(self.model)
        if params.ema:
            self.ema_model = EMA(self.model, alpha=params.ema_alpha)

        self.optim = make_optim(self.model, params.optim, split=params.split_params)

        # memo
        self.train_size = 50000
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_record = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)

        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ids = batch['ids']
        nys = batch['nys']
        xs = batch['weak']
        metric_ys = batch['tys']

        # 提取特征
        outputs = self.model.forward(xs)
        logits = outputs.logits
        targets = torch.softmax(logits, dim=-1)

        Lall = F.cross_entropy(logits, nys)
        meter.last.lr = self.lr_sche.apply(self.optim, self.global_steps)

        self.optim.zero_grad()
        Lall.backward()
        self.optim.step()

        with torch.no_grad():
            meter.mean.Atx = torch.eq(logits.argmax(dim=-1), metric_ys).float().mean()
            n_mask = nys != metric_ys
            if n_mask.any():
                meter.mean.Anx = torch.eq(logits.argmax(dim=1)[n_mask], nys[n_mask]).float().mean()

            meter.Lall = Lall
            false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [metric_ys != nys]
            true_pred = targets.gather(1, metric_ys.unsqueeze(dim=1)).squeeze()  # [metric_ys != nys]
            self.true_pred_mem[ids, self.eidx] = true_pred
            self.false_pred_mem[ids, self.eidx] = false_pred
            self.loss_record[ids, self.eidx] = F.cross_entropy(logits, nys, reduction='none')

        if params.ema:
            self.ema_model.step()
        return meter

    def on_train_epoch_end(self, trainer: Trainer, func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        true_f = self.exp.mk_ipath('true.pth')
        false_f = self.exp.mk_ipath('false.pth')
        loss_f = self.exp.mk_ipath('loss.pth')

        # if self.eidx % 10 == 0:
        torch.save(self.true_pred_mem, true_f)
        torch.save(self.false_pred_mem, false_f)
        torch.save(self.loss_record, loss_f)


main = partial(main, trainer_cls=CrossEntropyTrainer, params_cls=ParamsType)
