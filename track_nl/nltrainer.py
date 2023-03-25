"""
refer to
 - https://github.com/kuangliu/pytorch-cifar
"""
from typing import ClassVar, Union

from lumo.data.loader import DataLoaderType
from torch.utils.data import DataLoader

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, Record, MetricType, TrainStage

from datasets.nldataset import get_train_dataset, get_test_dataset
from models.module_utils import ModelParams
from datasets.dataset_utils import DataParams
import torch
import numpy as np


class NLParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.epoch = 200
        self.train.batch_size = 128
        self.val_size = 0
        self.optim = self.OPTIM.create_optim('SGD', lr=0.1, weight_decay=5e-4, momentum=0.9)
        self.split_params = True

        self.ema = False

        self.noisy_type = self.choice('symmetric', 'asymmetric')
        self.noisy_ratio = self.arange(0.8, 0, 0.95)
        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None

        self.record_matrix = True

    def iparams(self):
        super(NLParams, self).iparams()
        if self.get("c", None) is not None:
            self.from_yaml(self.get("c"))


ParamsType = NLParams


class NLTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f'set seed {params.seed}')

    def on_process_loader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoader, dm: DataModule,
                              stage: TrainStage, *args, **kwargs):

        def right_part():
            if params.sche_type == 'cos':
                return params.SCHE.Cos(
                    start=params.optim.lr, end=1e-5,
                    left=len(self.train_dataloader) * params.warmup_epochs,
                    right=len(self.train_dataloader) * params.epoch
                )
            else:
                from lumo.core.interp import PowerDecay2
                return PowerDecay2(
                    start=params.optim.lr,
                    schedules=[100 * len(self.train_dataloader), 200 * len(self.train_dataloader)],
                    gammas=[0.1, 0.1]
                )

        if stage.is_train():
            if params.warmup_epochs > 0:
                self.lr_sche = params.SCHE.List([
                    params.SCHE.Linear(
                        start=1e-7, end=params.optim.lr,
                        left=0,
                        right=len(self.train_dataloader) * params.warmup_epochs
                    ),
                    right_part(),
                ])
            else:
                self.lr_sche = right_part()
            self.logger.info('create learning scheduler')
            self.logger.info(self.lr_sche)

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=5).hook(self)
        callbacks.LoggerCallback(step_frequence=30, break_in=150).hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        xs0 = batch['xs']
        idx = batch['id']
        ys = batch['ys']
        logits = self.to_logits(xs0)
        pred_ys = logits.argmax(dim=-1)
        self.real_ys.extend(ys.cpu().numpy())
        self.pred_ys.extend(pred_ys.cpu().numpy())
        self.pred_idx.extend(idx.cpu().numpy())

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, dm: Union[DataModule, DataLoaderType] = None,
                       arg_params: ParamsType = None, *args, **kwargs):
        super().on_train_begin(trainer, func, params, dm, arg_params, *args, **kwargs)

    def on_test_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        self.pred_ys = []
        self.real_ys = []
        self.pred_idx = []

    def on_test_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        self.pred_idx = np.array(self.pred_idx)
        self.pred_ys = np.array(self.pred_ys)[self.pred_idx]
        self.real_ys = np.array(self.real_ys)[self.pred_idx]

        if self.is_main:
            from sklearn import metrics

            acc = metrics.accuracy_score(self.real_ys, self.pred_ys)
            if params.get("record_matrix", False):
                cm = metrics.confusion_matrix(self.real_ys,
                                              self.pred_ys,
                                              labels=np.arange(
                                                  params.n_classes))
                self.logger.raw(cm)
                max_acc = self.metric.dump_metric('Acc', acc, 'max', flush=True, cm=cm)
            else:
                max_acc = self.metric.dump_metric('Acc', acc, 'max', flush=True)
            self.logger.info(f'Best Acc: {max_acc}, Current: {acc}.')


class NLDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   noisy_ratio=params.noisy_ratio,
                                   method=params.method, noisy_type=params.noisy_type)

            if params.val_size > 0:
                eval_ds = get_train_dataset(params.dataset,
                                            noisy_ratio=params.noisy_ratio,
                                            method=params.method, noisy_type=params.noisy_type)

                idx = np.arange(len(ds))
                np.random.shuffle(idx)

                ds.subset(idx[params.val_size:]), eval_ds.subset(idx[:params.val_size])
                eval_dl = eval_ds.DataLoader(**params.eval.to_dict())
                self.regist_dataloader_with_stage(stage.val, eval_dl)

            dl = ds.DataLoader(**params.train.to_dict())

            self.regist_dataloader_with_stage(stage, dl)

        else:
            ds = get_test_dataset(params.dataset)
            dl = ds.DataLoader(**params.test.to_dict())
            self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = NLDM(params)
    trainer = trainer_cls(params, dm)  # type: Trainer

    if params.get('eval_first', False):
        trainer.test()

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_last_model()
