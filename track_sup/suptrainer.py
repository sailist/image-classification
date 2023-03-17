"""
refer to
 - https://github.com/kuangliu/pytorch-cifar
"""
from abc import ABC
from typing import ClassVar

from torch.utils.data import DataLoader

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, Record, MetricType, TrainStage

from datasets.supdataset import get_train_dataset, get_test_dataset
from models.module_utils import ModelParams
from datasets.dataset_utils import DataParams
import torch


class SupParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.epoch = 200
        self.batch_size = 128
        self.optim = self.OPTIM.create_optim('SGD', lr=0.1, weight_decay=5e-4, momentum=0.9)
        self.split_params = True

        self.ema = False

        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None


ParamsType = SupParams


class SupTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback, ABC):

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        self.rnd.mark(params.seed)
        self.logger.info(f'set seed {params.seed}')
        self.accuracy = []

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
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        xs0 = batch['xs0']
        xs1 = batch['xs1']
        ys = batch['ys']
        logits0 = self.to_logits(xs0)
        logits1 = self.to_logits(xs1)

        meter.sum.Acc = torch.eq(logits0.argmax(dim=-1), ys).sum()
        meter.sum.Acc1 = torch.eq(logits1.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def on_test_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        record.avg()
        self.accuracy.extend([record.avg()['Acc'], record.avg()['Acc1']])


class SupDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   method=params.method)

            dl = ds.DataLoader(**params.train.to_dict())

        else:
            ds = get_test_dataset(params.dataset)
            dl = ds.DataLoader(**params.test.to_dict())
        print(ds, stage)

        self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = SupDM(params)
    trainer = trainer_cls(params, dm) # type: Trainer

    if params.pretrain_path is not None and params.train_linear:
        trainer.load_state_dict(params.pretrain_path)
        trainer.test()
        return

    if params.get('eval_first', False):
        trainer.test()

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_last_model()
