"""
refer to
    - https://github.com/TorchSSL/TorchSSL
"""
from collections import Counter
from functools import partial

from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F

from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .semitrainer import *


class FlexMatchParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.epoch = 1024
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.03,
                                             momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)

        self.method = 'flexmatch'
        self.apply_da = False
        self.apply_flex = True  # if False, will be degenerated as fixmatch method.
        self.thresh_warmup = True  # if True, train all unlabeled data at start

        self.apply_nllce = True


ParamsType = FlexMatchParams


class FlexMatchModule(nn.Module):

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


class FlexMatchTrainer(SemiTrainer):

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = FlexMatchModule(params.model,
                                     n_classes=params.n_classes)

        self.optim = params.optim.build(self.model.parameters())
        self.tensors.register('flex_pys', -1, self.ds_size, dtype=torch.long)
        # if params.dataset in {'cifar10', 'cifar100'}:
        #     self.pseudo_label = torch.full((50000,), fill_value=-1,
        #                                    device=self.device, dtype=torch.long)
        # elif params.dataset in {'stl10'}:
        #     self.pseudo_label = torch.full((105000,), fill_value=-1,
        #                                    device=self.device, dtype=torch.long)
        # else:
        #     raise NotImplementedError()

        self.to_device()
        self.classwise_acc = None
        self.da_prob_list = []
        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def on_process_loader_end(self, trainer: "Trainer", func, params: ParamsType, loader: DataLoaderType,
                              dm: DataModule, stage: TrainStage, *args, **kwargs):
        super().on_process_loader_end(trainer, func, params, loader, dm, stage, *args, **kwargs)

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        xs, ys = batch_x['xs'], batch_x['ys']

        uids = batch_un['ids']
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

        un_probs, un_w_pys = torch.max(un_w_probs, dim=1)

        def calc_class_acc():
            classwise_acc = torch.zeros(params.n_classes, device=self.device, dtype=torch.float)
            pseudo_counter = Counter(self.tensors['flex_pys'].tolist())
            # if max(pseudo_counter.values()) < self.ds_size:  # not all(5w) -1
            if any(self.tensors['flex_pys'] != -1):
                if not params.thresh_warmup:  # do not consider the count number of -1
                    if -1 in pseudo_counter.keys():
                        pseudo_counter.pop(-1)

                for i in range(params.n_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            self.classwise_acc = classwise_acc
            return classwise_acc

        if params.apply_flex:
            classwise_acc = calc_class_acc()
        else:
            classwise_acc = torch.ones(params.n_classes, device=self.device, dtype=torch.float)

        # _pseudo_thresh = params.pseudo_thresh # fixmatch thresh
        # flexmatch thresh ↓
        _pseudo_thresh = params.pseudo_thresh * (classwise_acc[un_w_pys] / (2. - classwise_acc[un_w_pys]))
        p_mask = un_probs.ge(_pseudo_thresh)

        select_mask = un_probs.ge(params.pseudo_thresh)
        self.tensors.scatter('flex_pys', un_w_pys[select_mask], uids[select_mask])

        if params.apply_nllce:
            _log_pred = F.log_softmax(un_s0_logits, dim=-1)
            Lu = F.nll_loss(_log_pred, un_w_pys, reduction='none') * p_mask.float()
        else:
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

    def on_train_step_end(self, trainer: 'Trainer', func, params: ParamsType, metric: MetricType = None, *args,
                          **kwargs):
        super().on_train_step_end(trainer, func, params, metric, *args, **kwargs)
        if self.idx % 150 == 0:
            self.logger.info(self.classwise_acc)


main = partial(main, FlexMatchTrainer, ParamsType)
