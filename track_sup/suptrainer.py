"""
refer to
 - https://github.com/kuangliu/pytorch-cifar
"""
from typing import ClassVar
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, Record, MetricType, TrainStage

from datasets.supdataset import get_train_dataset, get_test_dataset
from models.module_utils import ModelParams
from datasets.dataset_utils import DataParams


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


class SupTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

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
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)
        # callbacks.EMAUpdate().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_ema_logits(self, xs):
        raise NotImplementedError()

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if logits.ndim == 3:
            meter.sum.Lall = F.cross_entropy(logits.permute(0, 2, 1), ys)
        else:
            meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if params.get('confuse_matrix', False):
            attention_mask = (ys >= 0)
            self.true.extend(ys[attention_mask].cpu().numpy().tolist())
            self.pred.extend(logits[attention_mask].argmax(dim=-1).cpu().numpy().tolist())

        if logits.ndim == 3:
            meter.sum.Lall = F.cross_entropy(logits.permute(0, 2, 1), ys)
        else:
            meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def on_eval_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_eval_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                self.pred = []
                self.true = []

    def on_test_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                self.pred = []
                self.true = []

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                from sklearn import metrics
                if len(self.pred) > 0:
                    cm = metrics.confusion_matrix(self.pred, self.true, labels=range(params.n_classes))
                    self.logger.raw(cm)

                    cls_pre, cls_rec, cls_f1, _ = metrics.precision_recall_fscore_support(
                        self.true, self.pred
                    )
                    # cls_pre = {k: v for k, v in zip(params.class_names, cls_pre)}
                    # cls_rec = {k: v for k, v in zip(params.class_names, cls_rec)}
                    # cls_f1 = {k: v for k, v in zip(params.class_names, cls_f1)}

                    accuracy = metrics.accuracy_score(self.true, self.pred)
                    wa = metrics.balanced_accuracy_score(self.true, self.pred)
                    precision = metrics.precision_score(self.true, self.pred, average='weighted')
                    recall = metrics.recall_score(self.true, self.pred, average='weighted')
                    wf1 = metrics.f1_score(self.true, self.pred, average='weighted')
                    mif1 = metrics.f1_score(self.true, self.pred, average='micro')
                    maf1 = metrics.f1_score(self.true, self.pred, average='macro')

                    m = Meter()

                    with self.database:
                        m.update(self.database.update_metric_pair('pre', precision, 'cls_pre', cls_pre, compare='max'))
                        m.update(self.database.update_metric_pair('rec', recall, 'cls_rec', cls_rec, compare='max'))
                        m.update(self.database.update_metric_pair('f1', wf1, 'cls_f1', cls_f1, compare='max'))
                        m.update(self.database.update_metrics(dict(acc=accuracy,
                                                                   wa=wa,
                                                                   mif1=mif1,
                                                                   maf1=maf1),
                                                              compare='max'))
                        self.database.flush()

                    self.logger.info('Best Results', m)


class SupDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   method=params.method)

            dl = ds.DataLoader(batch_size=params.batch_size, **params.train.to_dict())

        else:
            ds = get_test_dataset(params.dataset)
            dl = ds.DataLoader(batch_size=params.batch_size, **params.test.to_dict())
        print(ds, stage)

        self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = SupDM(params)
    trainer = trainer_cls(params, dm)

    if params.pretrain_path is not None and params.train_linear:
        trainer.load_state_dict(params.pretrain_path)
        trainer.test()
        return

    if params.get('eval_first', False):
        trainer.test()

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_model()
