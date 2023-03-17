import json
from typing import ClassVar, Union
from torch import nn
import torch
from lumo import TrainStage
from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, MetricType, Record
from lumo.utils.device import send_to_device
from lumo.data.loader import DataLoaderType
from torch.nn import functional as F

from contrib.load_ssl_model import SSLLoadModel
from datasets.dataset_utils import DataParams
from datasets.ssldataset import get_train_dataset, get_test_dataset
from models.knn import knn_predict
from models.memory_bank import StorageBank
from models.module_utils import ModelParams, ResnetOutput


class SSLParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.method = None

        self.epoch = 500
        self.train.batch_size = 512
        self.test.batch_size = 512
        self.eval.batch_size = 512

        self.optim = self.OPTIM.create_optim('Adam', lr=1e-3, weight_decay=1e-6)

        self.ema = True

        self.more_sample = True  # use more unlabeled data if has, like stl-10

        self.temperature = 0.5
        self.detach_cls = True
        self.feature_dim = 128
        self.hidden_feature_size = 128
        self.warmup_epochs = 10
        self.warmup_from = 0.01
        self.lr_decay_end = 0.0005
        # tricks
        ## Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning
        self.apply_unmix = False  # TODO
        ## MixCo: Mix-up Contrastive Learning for Visual Representation
        self.apply_mixco = False  # TODO

        # train a linear classifier or not
        self.train_linear = True
        self.train_strategy = self.choice(
            'ending',  # train on the last n epoch
            'always',  # train on the whole procedure
        )
        self.train_ending = 10
        self.pretrain_path = None

        self.knn = True
        self.knn_k = 200
        self.knn_t = 0.1

        self.linear_eval = False
        self.semi_eval = False

    def iparams(self):
        super().iparams()


ParamsType = SSLParams


class SSLTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.tensors = StorageBank()
        self.tensors.register('test_feature', dim=params.feature_dim, k=len(self.dm.test_dataset))
        self.tensors.register('test_ys', dim=-1, k=len(self.dm.test_dataset), dtype=torch.long)

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f'set seed {params.seed}')

    def on_process_loader_end(self, trainer: 'Trainer', func, params: ParamsType, loader: DataLoaderType,
                              dm: DataModule,
                              stage: TrainStage, *args, **kwargs):
        if stage.is_train():
            if params.warmup_epochs > 0:
                self.lr_sche = params.SCHE.List([
                    params.SCHE.Linear(
                        start=params.warmup_from, end=params.optim.lr,
                        left=0,
                        right=len(self.train_dataloader) * params.warmup_epochs
                    ),
                    params.SCHE.Cos(
                        start=params.optim.lr, end=params.lr_decay_end,
                        left=len(self.train_dataloader) * params.warmup_epochs,
                        right=len(self.train_dataloader) * params.epoch
                    ),
                ])
            else:
                self.lr_sche = params.SCHE.Cos(
                    start=params.optim.lr, end=params.lr_decay_end,
                    left=0,
                    right=len(self.train_dataloader) * params.epoch
                )
            self.logger.info('create learning scheduler')
            self.logger.info(self.lr_sche)

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=10).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        SSLLoadModel().hook(self)

        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_resnet_output(self, xs) -> ResnetOutput:
        if self.params.ema:
            return self.ema_model.forward(xs)
        return self.model.forward(xs)

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_feature(self, xs):
        raise NotImplementedError()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        idx = batch['id']
        xs0 = batch['xs0']
        xs1 = batch['xs1']
        ys = batch['ys']
        logits0 = self.to_logits(xs0)
        logits1 = self.to_logits(xs1)
        feature = self.to_feature(xs0)
        feature = F.normalize(feature, dim=-1)
        self.tensors.scatter('test_feature', feature, idx)
        self.tensors.scatter('test_ys', ys, idx)

        meter.sum.Acc0 = torch.eq(logits0.argmax(dim=-1), ys).sum()
        meter.sum.Acc1 = torch.eq(logits1.argmax(dim=-1), ys).sum()
        meter.sum.C = xs0.shape[0]
        return meter

    @property
    def metric_step(self):
        return self.global_steps % 100 == 0

    def on_train_begin(self, trainer: Trainer, func, params: ParamsType, dm: Union[DataModule, DataLoaderType] = None,
                       arg_params: ParamsType = None, *args, **kwargs):
        super().on_train_begin(trainer, func, params, dm, arg_params, *args, **kwargs)
        self.acc = 0

    def on_test_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        acc = record.agg()['Acc0']
        self.metric.dump_metric('Acc', acc, cmp='max')
        self.save_last_model()

        @torch.no_grad()
        def knn_test():
            self.change_stage(TrainStage.val)
            feature_bank = []
            with torch.no_grad():
                # generate feature bank
                for batch in self.dm['memo']:
                    batch = send_to_device(batch, self.device)
                    data, target = batch['xs'], batch['ys']
                    feature = self.to_feature(data)
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)

                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
                # [N]
                feature_labels = torch.tensor(self.dm['memo'].dataset.inputs['ys'], device=feature_bank.device)
                # loop test data to predict the label by weighted knn search

                pred_labels = knn_predict(self.tensors['test_feature'],
                                          feature_bank, feature_labels, params.n_classes, params.knn_k, params.knn_t)
                total_num = pred_labels.shape[0]
                total_top1 = torch.eq(pred_labels[:, 0], self.tensors['test_ys']).float().sum().item()

            self.change_stage(TrainStage.train)
            return total_top1 / total_num * 100

        if params.knn:
            knn_acc = knn_test()
            max_knn_acc = self.metric.dump_metric('Knn', knn_acc, cmp='max', flush=True)
            self.logger.info(f'Best Knn Top-1 acc: {max_knn_acc}, current: {knn_acc}')

            if knn_acc >= max_knn_acc:
                self.save_best_model()


class SSLDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)

        if stage.is_train():
            if params.dataset == 'stl10' and params.stl10_unlabeled:
                split = 'train+unlabeled'
            else:
                split = 'train'

            ds = get_train_dataset(params.dataset,
                                   method=params.method, split=split)

            dl = (
                ds.DataLoader(batch_size=params.train.batch_size,
                              num_workers=params.train.num_workers,
                              shuffle=True,
                              pin_memory=True, drop_last=True)
            )

            if params.dataset == 'stl10':
                ds = get_train_dataset(params.dataset,
                                       method=params.method, split='train')

            # used for knn-eval
            memo_dl = ds.DataLoader(batch_size=params.train.batch_size,
                                    num_workers=params.train.num_workers,
                                    drop_last=False, shuffle=False)
            self.regist_dataloader(memo=memo_dl)
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
    params = params_cls()  # type: ParamsType
    params.from_args()

    dm = SSLDM()
    trainer = trainer_cls(params, dm)  # type: SSLTrainer

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_last_model()

    if params.linear_eval:
        from track_ssl.linear import LinearEvalParams, LinearEvalTrainer
        eval_params = LinearEvalParams()
        eval_params.dataset = params.dataset
        eval_params.model = params.model
        eval_params.scan = 'linear-eval-' + params.get('scan', '')
        eval_params.pretrain_path = trainer.exp.mk_bpath('models', 'best_model.ckpt')

        eval_trainer = LinearEvalTrainer(eval_params, dm)
        eval_trainer.train()
        eval_trainer.test()
