"""
按照 p_mask 过滤 topk 的部分样本
"""
from functools import partial
from typing import Optional

from lumo.contrib import EMA
from lumo.contrib.nn.loss import contrastive_loss2
from torch.nn import functional as F
from models.components import MLP
from track_semi.semitrainer import *
from torch import nn
from models.module_utils import (pick_model_name,
                                 ResnetOutput)
from contrib import sharpen
from datasets.const import coarse_label_map
from models.memory_bank import batch_shuffle_ddp, batch_unshuffle_ddp


class CHMatchParams(SemiParams):

    def __init__(self):
        super().__init__()
        self.epoch = 1024
        self.method = 'chmatch'

        self.apply_da = True

        self.include_ssl = True

        self.apply_h_fc = True
        self.apply_h_mask = True

        self.c_classes = 20
        self.first_strong = True
        self.second_strong = False
        self.mlen = 50000

        self.fc = True
        self.w_cs = 1
        self.w_alpha = 1
        self.w_beta = 1
        self.w_gamma = 1

        self.k_type = self.choice('linear', 'cos', 'log', 'exp', 'fixed')
        self.min_proportion = 0.05
        self.max_proportion = 0.8
        self.k_inc_epoch = 100

        self.pp_rate = 0.05

        # False to degradation into FixMatch (Fixed threshold 0.95
        self.proportion_thresh = True

        self.moco_bank = False
        self.moco_bank_size = 65535

    def iparams(self):
        super().iparams()


ParamsType = CHMatchParams


class CoarseResnetOutput(ResnetOutput):
    c_logits: Optional[torch.Tensor] = None
    featureb: Optional[torch.Tensor] = None


class CHMatchModule(nn.Module):

    def __init__(self,
                 model_name,
                 n_classes=10,
                 c_classes=20,
                 with_coarse=True):
        super().__init__()
        self.backbone = pick_model_name(model_name)
        self.feature_dim = self.backbone.feature_dim
        self.head = MLP(feature_dim=self.feature_dim, output_dim=self.feature_dim, with_bn=True)
        self.classifier = nn.Linear(self.feature_dim, n_classes)
        self.wc = with_coarse
        if self.wc:
            self.c_classifier = nn.Linear(self.feature_dim, c_classes)

    def forward(self, xs):
        output = CoarseResnetOutput()
        feature_map = self.backbone(xs)

        feature = self.head(feature_map)

        logits = self.classifier(feature_map)

        if self.wc:
            c_logits = self.c_classifier(feature_map)
            output.c_logits = c_logits

        output.feature = feature
        output.feature_map = feature_map
        output.logits = logits
        return output


class CHMatchTrainer(SemiTrainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)

    def to_logits(self, xs):
        return self.model.forward(xs).logits

    def to_ema_logits(self, xs):
        if self.params.ema:
            return self.ema_model.forward(xs).logits
        return self.model.forward(xs).logits

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = CHMatchModule(params.model,
                                   n_classes=params.n_classes,
                                   c_classes=params.c_classes,
                                   with_coarse=params.fc,
                                   )
        self.optim = params.optim.build(self.model.parameters())

        if self.is_dist:
            self.logger.info('convert BatchNorm into SyncBatchNorm')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.mem.register('fine', -1, params.mlen)
        self.mem.register('coarse', -1, params.mlen)

        self.da_prob_list = []
        self.da_prob_c_list = []

        type_map = {
            'linear': params.SCHE.Linear,
            'cos': params.SCHE.Cos,
            'exp': params.SCHE.Exp,
            'log': params.SCHE.Log,
        }
        if params.k_type != 'fixed':
            sche_fn = type_map[params.k_type]
            self.k_sche = sche_fn(params.mlen * params.min_proportion,
                                  params.mlen * params.max_proportion,
                                  left=0, right=params.k_inc_epoch)
        else:
            self.k_sche = params.SCHE.Constant(params.mlen * params.max_proportion)

        if params.moco_bank:
            self.mem.register('strong', self.model.feature_dim, params.moco_bank_size)
            self.mem.register('prob_fine', params.n_classes, params.moco_bank_size)
            if params.fc:
                self.mem.register('prob_coarse', params.c_classes, params.moco_bank_size)

        if params.fc:
            fc_map = coarse_label_map[params.dataset]
            self.mmap = torch.tensor(
                [fc_map[i] for i in range(params.n_classes)],
                device=self.device,
                dtype=torch.long,
            )

        if params.ema or params.moco_bank:
            self.ema_model = EMA(self.model, alpha=0.999)

        ds_size = len(self.dm.train_dataset['un'])
        self.tensors.register('cpys', -1, ds_size, dtype=torch.long)
        self.tensors.register('cpscore', -1, ds_size)
        self.logger.info(f'register cpys: {ds_size}')
        self.logger.info(f'register cpscore: {ds_size}')

        self.to_device()

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        batch_x, batch_un = batch
        # labels
        idx = batch_x['id']
        idu = batch_un['id']
        ys = batch_x['ys']
        metric_uys = batch_un['ys']

        with torch.no_grad():
            if 'stl' not in params.dataset:
                self.tensors['tys'][idx] = ys
                self.tensors['tys'][idu] = metric_uys

        # images
        xs = batch_x['xs']
        uxs, uxs_s0, uxs_s1 = (
            batch_un['xs1'],
            batch_un['randaug'],
            batch_un['simclr']
        )

        sup_size = xs.shape[0]

        if params.moco_bank:
            output = self.model.forward(torch.cat([xs, uxs, uxs_s0]))
            imgs_m = torch.cat([xs, uxs, uxs_s1])
            imgs_m, idx_unshuffle = batch_shuffle_ddp(imgs_m)
            (x_w_logits), (un_w_logits, un_s0_logits) = (
                output.logits[:sup_size],
                output.logits[sup_size:].chunk(2)
            )

            m_output = self.ema_model.forward(imgs_m)

            m_output.logits = batch_unshuffle_ddp(m_output.logits, idx_unshuffle)
            m_output.feature = batch_unshuffle_ddp(m_output.feature, idx_unshuffle)
        else:
            output = self.model.forward(torch.cat([xs, uxs, uxs_s0, uxs_s1]))
            (x_w_logits), (un_w_logits, un_s0_logits, _) = (
                output.logits[:sup_size],
                output.logits[sup_size:].chunk(3)
            )

        @torch.no_grad()
        def data_align():
            probs = un_w_probs
            self.da_prob_list.append(self.accelerate.gather(probs).detach().mean(0))
            if len(self.da_prob_list) > 32:
                self.da_prob_list.pop(0)
            prob_avg = torch.stack(self.da_prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            return probs

        @torch.no_grad()
        def c_data_align():
            probs = un_w_c_probs
            self.da_prob_c_list.append(self.accelerate.gather(probs).detach().mean(0))
            if len(self.da_prob_c_list) > 32:
                self.da_prob_c_list.pop(0)
            prob_avg = torch.stack(self.da_prob_c_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            return probs

        # find dynamic threshold for fine grained labels
        with torch.no_grad():
            un_w_probs = torch.softmax(un_w_logits.detach(), dim=1)
            if params.apply_da:
                un_w_probs = data_align()

            scores, un_w_pys = torch.max(un_w_probs, dim=1)

            # Memory Bank for storing fine grained label prob
            if params.proportion_thresh:
                thresh, _ = self.mem['fine'].topk(int(self.k_sche(self.eidx)))
                self.mem.push('fine', self.accelerate.gather(scores))
                p_mask = scores.ge(thresh[-1])
                meter.last.t = thresh[-1]
            else:
                # fixmatch threshold
                p_mask = scores.ge(0.95)

            pp_mask = (~p_mask)

            meter.mean.ppm = pp_mask.float().mean()

        Lu = F.cross_entropy(un_s0_logits, un_w_pys, reduction='none') * p_mask.float()
        Lu = Lu.mean()
        Lx = F.cross_entropy(x_w_logits, ys)

        if params.record_predict and 'stl' not in params.dataset:
            with torch.no_grad():
                self.tensors.scatter('pscore', scores, idu)
                self.tensors.scatter('pys', un_w_pys, idu)

        Lcoarse = 0
        if params.fc and params.apply_h_fc:
            cys = self.mmap[ys]
            metric_cuys = self.mmap[metric_uys]
            if params.moco_bank:
                (x_w_c_logits), (un_w_c_logits, un_s0_c_logits) = (
                    output.c_logits[:sup_size],
                    output.c_logits[sup_size:].chunk(2)
                )
            else:
                (x_w_c_logits), (un_w_c_logits, un_s0_c_logits, _) = (
                    output.c_logits[:sup_size],
                    output.c_logits[sup_size:].chunk(3)
                )

            # find dynamic threshold for coarse grained labels
            with torch.no_grad():
                un_w_c_probs = torch.softmax(un_w_c_logits.detach(), dim=1)
                c_scores, un_w_c_pys = torch.max(un_w_c_probs, dim=1)

                if params.proportion_thresh:
                    cthresh, _ = self.mem['coarse'].topk(int(self.k_sche(self.eidx)))
                    self.mem.push('coarse', self.accelerate.gather(c_scores))
                    cp_mask = c_scores.ge(cthresh[-1])
                    meter.last.ct = cthresh[-1]
                else:
                    cp_mask = c_scores.ge(0.95)

            if params.apply_da:
                un_w_c_probs = c_data_align()

            Lcu = F.cross_entropy(un_s0_c_logits, un_w_c_pys, reduction='none') * cp_mask.float()
            Lcu = Lcu.mean()
            Lcx = F.cross_entropy(x_w_c_logits, cys)

            if params.record_predict and 'stl' not in params.dataset:
                with torch.no_grad():
                    self.tensors.scatter('cpscore', c_scores, idu)
                    self.tensors.scatter('cpys', un_w_c_pys, idu)

            Lcoarse = Lcu + Lcx

            with torch.no_grad():
                meter.mean.Lcu = Lcu
                meter.mean.Lcx = Lcx
                if cp_mask.any():
                    meter.mean.Acum = torch.eq(un_w_c_logits.argmax(dim=-1), metric_cuys)[cp_mask].float().mean()
                    meter.mean.cum = cp_mask.float().mean()

        if params.include_ssl:
            if params.moco_bank:
                (_), (_, un_s0_feature) = (
                    output.feature[:sup_size],
                    output.feature[sup_size:].chunk(2)
                )
                (_), (_, un_s1_feature) = (
                    m_output.feature[:sup_size],
                    m_output.feature[sup_size:].chunk(2)
                )
            else:
                (_), (_, un_s0_feature, un_s1_feature) = (
                    output.feature[:sup_size],
                    output.feature[sup_size:].chunk(3)
                )

            with torch.no_grad():
                qk_graph = torch.eq(un_w_pys[:, None], un_w_pys[None, :])
                if params.fc:
                    if params.apply_h_mask:
                        qkk_graph = un_w_c_pys[:, None] == un_w_c_pys[None, :]
                        qk_graph = qk_graph * qkk_graph

                qk_graph = qk_graph | torch.eye(len(qk_graph), device=self.device, dtype=torch.bool)
                qk_graph = sharpen(qk_graph, 1)

            if params.moco_bank:
                m_feat = self.mem['strong'].clone().detach()
                m_prob = self.mem['prob_fine'].clone().detach()
                m_cprob = self.mem['prob_coarse'].clone().detach()
                _, m_w_pys = torch.max(m_prob, dim=1)
                _, m_c_w_pys = torch.max(m_cprob, dim=1)

                self.mem.push('strong', un_s1_feature)
                self.mem.push('prob_fine', un_w_probs)
                if params.fc:
                    self.mem.push('prob_coarse', un_w_c_probs)

                qm_graph = torch.eq(un_w_pys[:, None], m_w_pys[None, :])
                qmm_graph = un_w_c_pys[:, None] == m_c_w_pys[None, :]
                qm_graph = qm_graph * qmm_graph

                q_graph = torch.cat([qk_graph, qm_graph], dim=1)
                q_graph = sharpen(q_graph, 1)
                qk_graph, qm_graph = q_graph[:, :qk_graph.shape[1]], q_graph[:, qk_graph.shape[1]:]

                Lcs = contrastive_loss2(un_s0_feature, un_s1_feature, m_feat,
                                        qk_graph=qk_graph, qm_graph=qm_graph,
                                        norm=True, temperature=0.1)
            else:
                Lcs = contrastive_loss2(un_s0_feature, un_s1_feature,
                                        qk_graph=qk_graph,
                                        norm=True, temperature=0.1)
        else:
            Lcs = 0

        Lall = (
                Lx +
                params.w_alpha * Lu +
                params.w_beta * Lcoarse +
                params.w_gamma * Lcs
        )

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()
        self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema or params.moco_bank:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lall = Lall
            meter.mean.Lcs = Lcs
            meter.mean.Lu = Lu
            meter.mean.Lcc = Lcoarse

            meter.mean.Ax = torch.eq(x_w_logits.argmax(dim=-1), ys).float().mean()
            meter.mean.Au = torch.eq(un_w_logits.argmax(dim=-1), metric_uys).float().mean()
            if p_mask.any():
                meter.mean.Aum = torch.eq(un_w_logits.argmax(dim=-1), metric_uys)[p_mask].float().mean()
                meter.mean.um = p_mask.float().mean()

        return meter


main = partial(main, CHMatchTrainer, ParamsType)

if __name__ == '__main__':
    main()
