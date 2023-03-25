"""
A simple idea -> split noisy and clean data from threshold (local) and BetaMixture model (global)
"""
from functools import partial

import torch
from contrib.make_optim import make_optim
from lumo.contrib import EMA
from torch import nn
from torch.nn import functional as F

from contrib.nmix import NLMixture
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from track_nl.nltrainer import *

from lumo.contrib.nn.loss import cross_entropy_with_targets
from lumo.contrib.torch.tensor import onehot


class BetaGLParams(NLParams):

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
        self.filter_ema = 0.999
        self.burnin = 2
        self.mix_burnin = 5
        self.targets_ema = 0.3
        self.pred_thresh = 0.85
        self.local_filter = True  # 局部的筛选方法
        self.feature_mean = False
        self.mixt_ema = True

    def iparams(self):
        super().iparams()


ParamsType = BetaGLParams


class BetaGLModule(nn.Module):

    def __init__(self, model_name,
                 n_classes=10):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        input_dim = self.backbone.feature_dim
        self.classifier = nn.Linear(input_dim, n_classes)
        self.head = nn.Linear(input_dim, input_dim)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        logits = self.classifier(feature_map)
        feature = self.head(feature_map)
        output = ResnetOutput()
        output.feature = feature
        output.feature_map = feature_map
        output.logits = logits
        return output


class BetaGLTrainer(NLTrainer):

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

        self.model = BetaGLModule(params.model,
                                  n_classes=params.n_classes)

        # self.model2 = deepcopy(self.model)
        if params.ema:
            self.ema_model = EMA(self.model, alpha=params.ema_alpha)

        self.optim = make_optim(self.model, params.optim, split=params.split_params)

        # memo
        self.train_size = 50000
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        # self.pred_mem_size = self.train_size // params.n_classes
        # self.pred_mem = torch.zeros(self.pred_mem_size, params.n_classes, params.epoch,
        #                             dtype=torch.float, device=self.device)
        # self.filter_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        # self.count = torch.zeros(self.train_size, device=self.device, dtype=torch.int)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_record = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)
        self.gmm_pred = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

        self.plabel_sche = params.SCHE.Cos(0.1, 1, right=params.epoch // 2)
        self.gmm_w_sche = params.SCHE.Log(0.5, 1, right=params.epoch // 2)
        self.erl_sche = params.SCHE.Exp(start=0.5, end=1, right=params.epoch // 2)
        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ids = batch['ids']
        nys = batch['nys']
        xs = batch['weak']
        axs = batch['strong']
        metric_ys = batch['tys']

        # 提取特征
        w_outputs = self.model.forward(xs)
        w_logits = w_outputs.logits

        outputs = self.model.forward(axs)
        logits = outputs.logits

        with torch.no_grad():
            # 基于全局+局部筛选得到当前样本的噪音权重
            preds = torch.softmax(logits, dim=1).detach()
            label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
            # 先假定所有都是干净样本
            fweight = torch.ones(w_logits.shape[0], dtype=torch.float, device=self.device)
            if params.local_filter:
                # 局部筛选
                weight = label_pred - self.target_mem[ids]
                # 当前的和过去的 1，对 < 0.5 的，如果概率值下降了，就惩罚
                # 对 > 0.5 的，给一个正激励，哪怕没学到，也接着学（将错就错，积重难返）
                # 对 < 0.5 的，给一个负激励，如果没学到，就暂时不学了
                # weight = weight + (0.25 / params.n_classes) * (2 * label_pred - 1)
                weight = weight + (label_pred * 0.5 - 0.25) / params.n_classes
                weight_mask = weight < 0
                meter.tl = weight_mask[metric_ys == nys].float().mean()
                meter.fl = weight_mask[metric_ys != nys].float().mean()

            if self.eidx >= params.burnin:
                if params.local_filter:
                    fweight[weight_mask] -= self.gmm_w_sche(self.eidx)
                fweight -= self.noisy_cls[ids]
                fweight = torch.relu(fweight)

                # self.filter_mem[ids] = fweight
                remain = ((fweight > 0.6) | (fweight < 0.4)).float()
                meter.rem = remain.mean()

        # update_mask = self.count[ids] < self.eidx
        # meter.um = update_mask.float().mean()

        with torch.no_grad():
            # 弱增广样本的 target 平滑
            raw_targets = torch.softmax(w_logits, dim=1)
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets
            values, p_labels = targets.max(dim=-1)
            # self.plabel_mem[ids[update_mask]] = targets[update_mask]

            mask = values > params.pred_thresh
            if mask.any():
                meter.Apc = torch.eq(p_labels[mask], metric_ys[mask]).float().mean()

            n_targets = onehot(nys, params.n_classes)
            p_targets = onehot(p_labels, params.n_classes)

        # n_targets = n_targets * (1 - ratio) + p_targets * ratio
        # p_targets[mask.logical_not()] = p_targets.scatter(-1, top_indices[:, 1:2], ratio)[mask.logical_not()]

        mask = mask.float()
        meter.pm = mask.mean()

        Lce = cross_entropy_with_targets(logits, n_targets, fweight)
        Lpce = (cross_entropy_with_targets(logits, p_targets, (1 - fweight) | mask)
                * self.plabel_sche(self.eidx))

        Lall = Lce + Lpce

        meter.tw = fweight[metric_ys == nys].mean()
        meter.fw = fweight[metric_ys != nys].mean()
        meter.last.lr = self.lr_sche.apply(self.optim, self.global_steps)

        if params.local_filter:
            with torch.no_grad():
                # ids_mask = weight_mask.logical_not() * update_mask
                # 取局部筛选认为是正样本的，
                ids_mask = weight_mask.logical_not()
                # alpha = self.filter_ema_sche(self.eidx)
                alpha = params.filter_ema
                if self.eidx < params.burnin:
                    alpha = 0.99
                try:
                    self.target_mem[ids[ids_mask]] = (
                            self.target_mem[ids[ids_mask]] * alpha
                            + label_pred[ids_mask] * (1 - alpha))
                except IndexError as e:
                    print(ids)
                    print(ids_mask)
                    return meter

        self.optim.zero_grad()
        Lall.backward()
        self.optim.step()

        with torch.no_grad():
            meter.mean.Atx = torch.eq(logits.argmax(dim=-1), metric_ys).float().mean()
            n_mask = torch.eq(nys, metric_ys)
            if n_mask.any():
                meter.mean.Anx = torch.eq(logits.argmax(dim=1)[n_mask], nys[n_mask]).float().mean()

            meter.Lall = Lall
            meter.Lce = Lce
            meter.Lpcs = Lpce
            false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [metric_ys != nys]
            true_pred = targets.gather(1, metric_ys.unsqueeze(dim=1)).squeeze()  # [metric_ys != nys]
            self.true_pred_mem[ids, self.eidx] = true_pred
            self.false_pred_mem[ids, self.eidx] = false_pred
            # print(false_pred)
            self.loss_record[ids, self.eidx] = F.cross_entropy(w_logits, nys, reduction='none')
            # mem_mask = ids < self.pred_mem_size - 1
            # self.pred_mem[ids[mem_mask], :, self.eidx - 1] = targets[ids[mem_mask]]
            # self.count[ids[update_mask]] += 1

        if self.eidx == 1:
            self.cls_mem[ids, 0] = metric_ys
        elif self.eidx == 2:
            self.cls_mem[ids, 1] = nys
        else:
            self.cls_mem[ids, self.eidx - 1] = p_labels
        if params.ema:
            self.ema_model.step()
        return meter

    def on_train_epoch_end(self, trainer: Trainer, func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        true_f = self.exp.mk_ipath('true.pth')
        false_f = self.exp.mk_ipath('false.pth')
        loss_f = self.exp.mk_ipath('loss.pth')
        gmmpred_f = self.exp.mk_ipath('gmm.pth')

        # if self.eidx % 10 == 0:
        torch.save(self.true_pred_mem, true_f)
        torch.save(self.false_pred_mem, false_f)
        torch.save(self.loss_record, loss_f)
        torch.save(self.gmm_pred, gmmpred_f)

        nlmodel = NLMixture()
        with torch.no_grad():
            if self.eidx - 1 <= 0:
                self.logger.info('Skip global calc because of first epoch.')
                return

            f_mean = self.false_pred_mem[:, :self.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, self.eidx - 1].cpu().numpy()
            feature = nlmodel.create_feature(f_mean, f_cur)

            noisy_cls = nlmodel.bmm_predict(feature, mean=params.feature_mean, offset=0)
            if self.eidx > 1:
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
                true_cls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
                m = nlmodel.acc_mixture_(true_cls, self.noisy_cls_mem.cpu().numpy())
                self.logger.info(m)

                if self.eidx > params.mix_burnin:
                    if params.mixt_ema:
                        self.noisy_cls = self.noisy_cls_mem.clone()
                        self.gmm_pred[:, self.eidx] = torch.tensor(noisy_cls, device=self.device)
                    else:
                        self.noisy_cls = torch.tensor(noisy_cls, device=self.device)
                        self.gmm_pred[:, self.eidx] = torch.tensor(noisy_cls, device=self.device)


main = partial(main, trainer_cls=BetaGLTrainer, params_cls=ParamsType)
