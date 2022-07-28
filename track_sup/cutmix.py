"""

"""
import numpy as np
from contrib.cutmix import rand_bbox
from .mixup import *


class CutmixTrainer(MixupTrainer):

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()

        if params.aug_type == 'basic':
            xs = batch['xs']
        elif params.aug_type == 'simclr':
            xs = batch['sxs0']
        elif params.aug_type == 'randaug':
            xs = batch['sxs1']
        else:
            raise NotImplementedError()

        ys = batch['ys']
        target = onehot(ys, params.n_classes)

        if params.apply_cutmix:
            lam = np.random.beta(params.beta, params.beta)
            reid = torch.randperm(xs.size()[0], device=self.device)
            target_b = target[reid]
            bbx1, bby1, bbx2, bby2 = rand_bbox(xs.size(), lam)
            xs[:, :, bbx1:bbx2, bby1:bby2] = xs[reid, :, bbx1:bbx2, bby1:bby2]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xs.size()[-1] * xs.size()[-2]))
            output = self.model.forward(xs)

            if params.mixup_type == 'dual':
                Lx = (F.cross_entropy(output.logits, ys) * lam +
                      F.cross_entropy(output.logits, ys[reid]) * (1 - lam))
            elif params.mixup_type == 'multihot':
                mixed_targets = target * lam + target_b[reid] * (1 - lam)
                Lx = cross_entropy_with_targets(output.logits, mixed_targets)
            else:
                raise NotImplementedError()
        else:
            output = self.model.forward(xs)
            Lx = F.cross_entropy(output.logits, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lx)
        self.optim.step()
        cur_lr = self.lr_sche.apply(self.optim, self.global_steps)
        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.mean.Lx = Lx
            if params.apply_mixup:
                logits = self.model.forward(xs).logits
            else:
                logits = output.logits
            meter.mean.Ax = (logits.argmax(dim=-1) == ys).float().mean()
            meter.last.lr = cur_lr

        return meter


main = partial(main, trainer_cls=CutmixTrainer, params_cls=ParamsType)
