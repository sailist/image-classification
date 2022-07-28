import os
import json
from typing import ClassVar

import numpy as np
import torch
from dbrecord import PList
from torch.utils.data import RandomSampler

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, Record, DataLoaderSide, TrainStage, MetricType

from datasets.dataset_utils import DataParams
from datasets.semidataset import get_train_dataset, get_test_dataset
from lumo.data.loader import DataLoaderType
from lumo.proc.dist import is_main, is_dist, world_size
from models.module_utils import ModelParams


class SemiParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.method = None
        self.batch_count = 1024
        self.epoch = 1024
        self.batch_size = 64
        self.train.batch_size = 64
        self.test.batch_size = 100  # for multiprocess training
        self.optim = self.OPTIM.create_optim('SGD',
                                             lr=0.03,
                                             momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)
        self.model = 'wrn282'
        self.ema = True
        self.n_percls = 4

        self.num_workers = 8

        self.mu = 7
        self.pseudo_thresh = 0.95

        self.sharpen = 0.5
        self.mixup_beta = 0.5

        self.pretrain = True
        self.pretrain_path = None
        self.confuse_matrix = True

        self.record_predict = True

    def iparams(self):
        super().iparams()
        if self.dataset == 'cifar100':
            self.optim.weight_decay = 0.001


ParamsType = SemiParams


class SupTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.save_model()
            self.logger.info(f'set seed {params.seed}')

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        super().on_train_begin(trainer, func, params, *args, **kwargs)
        import psutil
        self.logger.info(psutil.Process(os.getppid()))

        float = torch.float
        if self.accelerate.mixed_precision == 'fp16':
            float = torch.float16

        if 'tinyimagenet' in params.dataset:
            self.pys = torch.full((100000,), -1, device=self.device, dtype=torch.long)
            self.cpys = torch.full((100000,), -1, device=self.device, dtype=torch.long)
            self.pscore = torch.full((100000,), -1, device=self.device, dtype=float)
            self.cpscore = torch.full((100000,), -1, device=self.device, dtype=float)
            self.tys = torch.full((100000,), -1, device=self.device, dtype=torch.long)
        else:
            self.pys = torch.full((50000,), -1, device=self.device, dtype=torch.long)
            self.cpys = torch.full((50000,), -1, device=self.device, dtype=torch.long)
            self.pscore = torch.full((50000,), -1, device=self.device, dtype=float)
            self.cpscore = torch.full((50000,), -1, device=self.device, dtype=float)
            self.tys = torch.full((50000,), -1, device=self.device, dtype=torch.long)

        ptrain = self.exp.blob_file('predicts.sqlite')
        self.logger.info(ptrain)
        self.predicts = PList(ptrain)

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        if is_main():
            if params.record_predict:
                with torch.no_grad():
                    self.predicts.append({'train':
                                              [self.pys.detach().tolist(), self.pscore.detach().tolist()]
                                          })
                self.predicts.flush()

    def on_process_loader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoaderType,
                              dm: DataModule,
                              stage: TrainStage, *args, **kwargs):
        if stage.is_train():
            self.lr_sche = params.SCHE.Cos(
                start=params.optim.lr,
                end=params.optim.lr * 0.2,
                left=0,
                right=len(self.train_dataloader) * 1024
            )
            self.logger.info(f'apply {self.lr_sche}')

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        self.accuracy = 0
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        confusion = self.exp.blob_file('confusion.sqlite')
        self.logger.info(confusion)
        self.plist = PList(confusion)

        callbacks.TensorBoardCallback().hook(self)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.AutoLoadModel().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_ema_logits(self, xs):
        return None

    def on_test_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        self.pred_ys = np.zeros(10000, dtype=np.long)
        self.pred_score = np.zeros(10000)
        self.real_ys = np.zeros(10000, dtype=np.long)

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        if is_main():
            if params.confuse_matrix:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(self.pred_ys, self.real_ys, labels=np.arange(params.n_classes))
                self.logger.raw(cm)
                self.plist.append(cm)
                self.plist.flush()

            if self.accuracy < record.agg()['Acc']:
                self.accuracy = record.agg()['Acc']
                self.save_best_model()
                self.save_last_model()

            if params.record_predict:
                self.predicts.append({'test': [self.pred_ys, self.pred_score]})
                self.predicts.flush()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        batch = self.to_device(batch)
        ids = batch['id']
        xs = batch['xs']
        ys = batch['ys']

        all_ys = self.accelerate.gather(ys)
        all_id = self.accelerate.gather(ids)
        if params.ema:
            logits = self.to_ema_logits(xs)
            nlogits = self.to_logits(xs)
            all_nlogits = self.accelerate.gather(nlogits)
            meter.sum.Anc = (all_nlogits.argmax(dim=-1) == all_ys).sum()
        else:
            logits = self.to_logits(xs)

        all_logits = self.accelerate.gather(logits)
        all_preds = torch.softmax(all_logits, dim=-1)
        all_pred_score, all_pred_ys = all_preds.max(dim=-1)
        if params.confuse_matrix:
            self.pred_ys[all_id.cpu().numpy()] = all_pred_ys.cpu().numpy()
            self.real_ys[all_id.cpu().numpy()] = all_ys.cpu().numpy()
            self.pred_score[all_id.cpu().numpy()] = all_pred_score.cpu().numpy()

        meter.sum.Acc = (all_logits.argmax(dim=-1) == all_ys).sum()
        meter.sum.C = all_ys.shape[0]
        return meter

    def save_best_model(self):
        file = self.exp.blob_file('best_model.ckpt', 'models')
        file_info = self.exp.blob_file('best_model.json', 'models')
        torch.save(self.state_dict(), file)
        with open(file_info, 'w') as w:
            w.write(json.dumps({'global_steps': self.global_steps, 'accuracy': self.accuracy}))
        self.logger.info(f'saved best model at {file}')

    def save_last_model(self):
        file = self.exp.blob_file('last_model.ckpt', 'models')
        file_info = self.exp.blob_file('last_model.json', 'models')
        torch.save(self.state_dict(), file)
        with open(file_info, 'w') as w:
            w.write(json.dumps({'global_steps': self.global_steps, 'accuracy': self.accuracy}))
        self.logger.info(f'saved last model at {file}')


class SemiDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)

        if stage.is_train():
            sup_ds, un_ds = get_train_dataset(params.dataset,
                                              n_percls=params.n_percls,
                                              method=params.method)

            sup_dl = sup_ds.DataLoader(batch_size=params.train.batch_size,
                                       num_workers=params.train.num_workers,
                                       sampler=RandomSampler(data_source=sup_ds,
                                                             replacement=True,
                                                             num_samples=params.batch_count * params.batch_size),
                                       pin_memory=True)
            un_dl = (
                un_ds.DataLoader(batch_size=params.train.batch_size * params.mu,
                                 num_workers=params.train.num_workers,
                                 sampler=RandomSampler(data_source=un_ds,
                                                       replacement=True,
                                                       num_samples=params.batch_count * params.batch_size * params.mu),
                                 pin_memory=True)
            )

            dl = DataLoaderSide().add('sup', sup_dl, cycle=True).add('un', un_dl).chain()
        elif stage.is_test():
            ds = get_test_dataset(params.dataset)

            dl = (
                ds.DataLoader(batch_size=params.test.batch_size,
                              num_workers=params.test.num_workers,
                              pin_memory=True, drop_last=False)
            )
        else:
            raise NotImplementedError()
        self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = SemiDM(params)
    trainer = trainer_cls(params, dm)

    if params.pretrain_path is not None:
        trainer.test()
        return

    trainer.train()
    trainer.save_model()
