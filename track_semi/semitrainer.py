import json
from typing import ClassVar

import numpy as np
import torch
from dbrecord import PList
from lumo import (
    Trainer,
    TrainerParams,
    Meter,
    callbacks,
    DataModule,
    Record,
    DataLoaderSide,
    TrainStage,
    MetricType,
)
from lumo.data.loader import DataLoaderType
from lumo.proc.dist import is_main
from torch.utils.data import RandomSampler

from contrib.load_ssl_model import SSLLoadModel
from datasets.dataset_utils import DataParams
from datasets.semidataset import get_train_dataset, get_test_dataset
from models.memory_bank import MemoryBank, StorageBank
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
        self.optim = self.OPTIM.create_optim("SGD",
                                             lr=0.03,
                                             momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)
        self.model = "wrn282"
        self.ema = True
        self.n_percls = 4

        self.mu = 7
        self.pseudo_thresh = 0.95

        self.sharpen = 0.5
        self.mixup_beta = 0.5

        self.pretrain = True
        self.pretrain_path = None

        # extra metric
        self.record_matrix = True

        # metric recording
        self.record_predict = True

        self.with_coarse = False

    def iparams(self):
        super().iparams()
        if self.dataset == "cifar100":
            self.optim.weight_decay = 0.001


ParamsType = SemiParams


class SemiTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    @classmethod
    def generate_exp_name(cls) -> str:
        return f"{cls.dirname()}.{cls.filebasename()}"

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.mem = MemoryBank()
        self.tensors = StorageBank()

        ds_size = len(self.dm.train_dataset['un'])
        self.tensors.register('pys', -1, ds_size, dtype=torch.long)
        self.tensors.register('pscore', -1, ds_size)
        self.tensors.register('tys', -1, ds_size, dtype=torch.long)

        test_ds_size = len(self.dm.test_dataset)
        self.tensors.register('test_pys', -1, test_ds_size, dtype=torch.long)
        self.tensors.register('test_ys', -1, test_ds_size, dtype=torch.long)
        self.tensors.register('test_pscore', -1, test_ds_size)

    def on_process_loader_begin(
            self,
            trainer: "Trainer",
            func,
            params: ParamsType,
            dm: DataModule,
            stage: TrainStage,
            *args,
            **kwargs,
    ):
        super().on_process_loader_begin(trainer, func, params, dm, stage,
                                        *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f"set seed {params.seed}")

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, *args,
                       **kwargs):
        super().on_train_begin(trainer, func, params, *args, **kwargs)

        ptrain = self.exp.mk_bpath("predicts.sqlite")
        self.logger.info(ptrain)
        self.predicts = PList(ptrain)

    def on_train_epoch_end(
            self,
            trainer: "Trainer",
            func,
            params: ParamsType,
            record: Record,
            *args,
            **kwargs,
    ):
        super().on_train_epoch_end(trainer, func, params, record, *args,
                                   **kwargs)
        if is_main():
            if params.record_predict:
                with torch.no_grad():
                    self.predicts.append({
                        "train": [
                            self.tensors['pys'].detach().tolist(),
                            self.tensors['pscore'].detach().tolist(),
                        ]
                    })
                self.predicts.flush()
            m = Meter()
            m.update(self.metric.value)
            self.logger.info(f'Metrics: {m}')

    def on_process_loader_end(
            self,
            trainer: "Trainer",
            func,
            params: ParamsType,
            loader: DataLoaderType,
            dm: DataModule,
            stage: TrainStage,
            *args,
            **kwargs,
    ):
        if stage.is_train():
            self.lr_sche = params.SCHE.Cos(
                start=params.optim.lr,
                end=params.optim.lr * 0.2,
                left=0,
                right=len(self.train_dataloader) * 1024,
            )
            self.logger.info(f"apply {self.lr_sche}")

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        confusion = self.exp.mk_bpath("confusion.sqlite")
        self.logger.info(confusion)
        self.plist = PList(confusion)

        # callbacks.TensorBoardCallback().hook(self)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        SSLLoadModel().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_ema_logits(self, xs):
        return None

    def on_test_end(
            self,
            trainer: "Trainer",
            func,
            params: ParamsType,
            record: Record = None,
            *args,
            **kwargs,
    ):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        pred_ys = np.array(self.tensors['test_pys'].cpu().numpy())
        real_ys = np.array(self.tensors['test_ys'].cpu().numpy())
        pred_score = np.array(self.tensors['test_pscore'].cpu().numpy())

        if is_main():
            from sklearn import metrics

            acc = metrics.accuracy_score(real_ys, pred_ys)
            if params.get("record_matrix", False):
                cm = metrics.confusion_matrix(real_ys,
                                              pred_ys,
                                              labels=np.arange(
                                                  params.n_classes))
                self.logger.raw(cm)
                self.plist.append(cm)
                self.plist.flush()
                max_acc = self.metric.dump_metric('Acc', acc, cmp='max', flush=True, cm=cm)
            else:
                max_acc = self.metric.dump_metric('Acc', acc, cmp='max', flush=True)

            self.logger.info(f'Best Acc: {max_acc}, Current: {acc}')

            if acc >= max_acc:
                self.save_best_model()
            self.save_last_model()

            if params.record_predict:
                self.predicts.append({"test": [pred_ys.tolist(), pred_score.tolist()]})
                self.predicts.flush()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        batch = self.to_device(batch)
        ids = batch["id"]
        xs = batch["xs"]
        ys = batch["ys"]

        all_ys = self.accelerate.gather(ys)
        all_id = self.accelerate.gather(ids)
        if params.ema:
            logits = self.to_ema_logits(xs)
            nlogits = self.to_logits(xs)
            all_nlogits = self.accelerate.gather(nlogits)
            meter.sum.Anc = torch.eq(all_nlogits.argmax(dim=-1), all_ys).sum()
        else:
            logits = self.to_logits(xs)

        all_logits = self.accelerate.gather(logits)
        all_preds = torch.softmax(all_logits, dim=-1)
        all_pred_score, all_pred_ys = all_preds.max(dim=-1)

        self.tensors.scatter('test_ys', all_ys, all_id)
        self.tensors.scatter('test_pys', all_pred_ys, all_id)
        self.tensors.scatter('test_pscore', all_pred_score, all_id)

        meter.sum.Acc = torch.eq(all_logits.argmax(dim=-1), all_ys).sum()
        meter.sum.C = all_ys.shape[0]
        return meter


class SemiDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)

        if stage.is_train():
            if params.dataset == 'stl10' and params.stl10_unlabeled:
                split = 'train+unlabeled'
            else:
                split = 'train'

            sup_ds, un_ds = get_train_dataset(params.dataset,
                                              n_percls=params.n_percls,
                                              method=params.method, split=split)

            sup_dl = sup_ds.DataLoader(
                batch_size=params.train.batch_size,
                num_workers=params.train.num_workers,
                sampler=RandomSampler(
                    data_source=sup_ds,
                    replacement=True,
                    num_samples=params.batch_count * params.batch_size if params.batch_count > 0 else None,
                ),
                pin_memory=True,
            )
            un_dl = un_ds.DataLoader(
                batch_size=params.train.batch_size * params.mu,
                num_workers=params.train.num_workers,
                sampler=RandomSampler(
                    data_source=un_ds,
                    replacement=True,
                    num_samples=params.batch_count * params.batch_size * params.mu if params.batch_count > 0 else None,
                ),
                pin_memory=True,
            )

            dl = (DataLoaderSide().add("sup", sup_dl,
                                       cycle=True).add("un", un_dl).chain())
        elif stage.is_test():
            ds = get_test_dataset(params.dataset)

            dl = ds.DataLoader(
                batch_size=params.test.batch_size,
                num_workers=params.test.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            raise NotImplementedError()
        self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = SemiDM(params)
    trainer = trainer_cls(params, dm)  # type: Trainer

    trainer.train()
    trainer.save_last_model()
