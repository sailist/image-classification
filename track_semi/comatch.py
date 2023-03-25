"""
CoMatch: Semi-supervised Learning with Contrastive Graph Regularization

refer to
 - https://github.com/salesforce/CoMatch
"""
import math
from functools import partial

from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from lumo.contrib.torch.tensor import onehot
from torch import nn
from torch.nn import functional as F

from contrib.tensor import sharpen
from models.components import MLP
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from .semitrainer import *


class CoMatchParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.epoch = 512
        self.method = 'comatch'
        self.model = 'wrn282'
        self.feature_dim = 64
        self.queue_size = 10
        self.temperature = 0.2

        self.apply_da = True
        self.apply_cs = True
        self.apply_avg = True
        self.apply_weak_graph = True
        self.with_bn = False
        self.alpha = 0.9
        self.contrast_th = 0.8

        self.w_lcs = 1  # weight for contrastive loss

    def iparams(self):
        super().iparams()
        if self.dataset == 'cifar100':
            self.model = 'wrn288'
        elif self.dataset == 'stl10':
            self.model = 'resnet18'
            self.w_lcs = 5


ParamsType = CoMatchParams


class CoMatchModule(nn.Module):

    def __init__(self, model_name,
                 feature_dim=64,
                 n_classes=10,
                 with_bn=False):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        self.backbone.apply(init_weight)
        input_dim = self.backbone.feature_dim
        self.feature_dim = feature_dim
        self.head = MLP(input_dim,
                        input_dim,
                        output_dim=feature_dim,
                        with_bn=with_bn)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, xs):
        feature_map = self.backbone(xs)
        feature = self.head(feature_map)
        logits = self.classifier(feature_map)

        output = ResnetOutput()
        output.feature_map = feature_map
        output.feature = feature
        output.logits = logits
        return output


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class CoMatchTrainer(SemiTrainer):

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.rnd.mark(params.seed)
        self.model = CoMatchModule(params.model,
                                   feature_dim=params.feature_dim,
                                   with_bn=params.with_bn,
                                   n_classes=params.n_classes)
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)

        param_list = [
            {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        self.optim = params.optim.build(param_list)

        self.optim = params.optim.build(self.model.parameters())

        self.da_prob_list = []
        # self.mb_feature = (MemoryBank(params.batch_size * params.mu * params.queue_size,
        #                               params.feature_dim).to(self.accelerator.device))

        self.mem.register('mb_feature',
                          dim=params.feature_dim,
                          k=params.batch_size * params.mu * params.queue_size)
        self.mem.register('mb_prob',
                          dim=params.n_classes,
                          k=params.batch_size * params.mu * params.queue_size)

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        xs, ys = batch_x['xs'], batch_x['ys']
        uxs, uxs_s0, uxs_s1 = batch_un['xs'], batch_un['sxs0'], batch_un['sxs1']
        metric_uys = batch_un['ys']

        sup_size = xs.shape[0]

        axs = torch.cat([xs, uxs, uxs_s0, uxs_s1])
        output = self.model.forward(axs)
        (x_w_logits), (un_w_logits, un_s0_logits, un_s1_logtis) = (
            output.logits[:sup_size],
            output.logits[sup_size:].chunk(3)
        )

        output.feature = F.normalize(output.feature, p=2, dim=-1)
        (x_w_featrure), (un_w_featrure, un_s0_featrure, un_s1_featrure) = (
            output.feature[:sup_size],
            output.feature[sup_size:].chunk(3)
        )

        @torch.no_grad()
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
        un_w_probs_ori = un_w_probs.clone()

        @torch.no_grad()
        def smooth_prob():
            A = torch.exp(torch.mm(un_w_featrure, self.mem['mb_feature'].t()) / params.temperature)
            A = A / A.sum(1, keepdim=True)
            probs = params.alpha * un_w_probs + (1 - params.alpha) * torch.mm(A, self.mem['mb_prob'])

            self.mem.push('mb_feature', torch.cat([un_w_featrure, x_w_featrure]))
            self.mem.push('mb_prob', torch.cat([un_w_probs_ori, onehot(ys, params.n_classes)]))
            return probs

        if params.apply_avg:
            un_w_probs = smooth_prob()

        scores, un_w_pys = torch.max(un_w_probs, dim=1)
        p_mask = scores.ge(params.pseudo_thresh)

        @torch.no_grad()
        def weak_graph():
            qk_graph = torch.mm(un_w_probs, un_w_probs.t())
            qk_graph.fill_diagonal_(1)
            pos_mask = (qk_graph >= params.contrast_th).float()
            qk_graph = qk_graph * pos_mask
            qk_graph = sharpen(qk_graph, T=1)
            return qk_graph

        Lcs = 0
        Q = None
        if params.apply_cs:
            if params.apply_weak_graph:
                Q = weak_graph()

            Lcs = contrastive_loss2(query=un_s0_featrure, key=un_s1_featrure,
                                    temperature=params.temperature,
                                    norm=False,
                                    qk_graph=Q,
                                    eye_one_in_qk=False) * params.w_lcs

        Lu = - torch.sum((F.log_softmax(un_s0_logits, dim=1) * un_w_probs), dim=1) * p_mask.float()
        Lu = Lu.mean()

        Lx = F.cross_entropy(x_w_logits, ys)

        Lall = Lx + Lu + Lcs

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
            meter.mean.Lcs = Lcs
            meter.mean.Ax = torch.eq(x_w_logits.argmax(dim=-1), ys).float().mean()
            meter.mean.Au = torch.eq(un_w_logits.argmax(dim=-1), metric_uys).float().mean()
            if p_mask.any():
                meter.mean.Aum = torch.eq(un_w_logits.argmax(dim=-1), metric_uys)[p_mask].float().mean()
                meter.mean.um = p_mask.float().mean()
            if Q is not None:
                label_graph = metric_uys.unsqueeze(0) == metric_uys.unsqueeze(1)
                Q.fill_diagonal_(0)
                label_graph.fill_diagonal_(0)
                Acm = (((Q > 0) * label_graph).sum(1) / (((Q > 0)).sum(1) + 1e-10))
                if (Acm > 0).any():
                    meter.mean.Acm = Acm[(Acm > 0)].mean()

        return meter


main = partial(main, CoMatchTrainer, ParamsType)
