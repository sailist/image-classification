from typing import ClassVar

import torch
from torch import nn
from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, MetricType, Record
from lumo import TrainStage

from models.module_utils import ModelParams, ResnetOutput
from datasets.ssldataset import get_train_dataset, get_test_dataset
from datasets.dataset_utils import DataParams


class SSLParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.method = None

        self.epoch = 500
        self.batch_size = 512
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

    def iparams(self):
        super().iparams()
        if self.dataset == 'stl10' and self.more_sample:
            self.train_linear = False


ParamsType = SSLParams


class SupTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

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
                        left=1e-3,
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
                    left=1e-3,
                    right=len(self.train_dataloader) * params.epoch
                )
            self.logger.info('create learning scheduler')
            self.logger.info(self.lr_sche)

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)
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

    def to_feature_any_logits(self, xs):
        raise NotImplementedError()

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        xs0 = batch['xs0']
        xs1 = batch['xs1']
        ys = batch['ys']
        logits0 = self.to_logits(xs0)
        logits1 = self.to_logits(xs1)

        meter.sum.Acc0 = (logits0.argmax(dim=-1) == ys).sum()
        meter.sum.Acc1 = (logits1.argmax(dim=-1) == ys).sum()
        meter.sum.C = xs0.shape[0]
        return meter

    @property
    def metric_step(self):
        return self.global_step % 100 == 0

    def save_backbone(self):
        model = getattr(self, 'model', None)
        backbone = None  # type:nn.Module
        if model is not None:
            backbone = getattr(self.model, 'backbone', None)
        if backbone is None:
            self.logger.info('cannot get backbone from self.model')
            return
        sd = backbone.state_dict()
        file = self.exp.blob_file(f'backbone_{self.global_step:06d}.pth', 'backbone_ckpts')
        torch.save(sd, file)
        self.logger.info(f'backbone saved in {file}')
        return file

    def on_train_epoch_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record, *args, **kwargs):
        if params.eidx % 100 == 0:
            self.save_model()
            self.save_backbone()


class SSLDM(DataModule):

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        super().idataloader(params, stage)

        if stage.is_train():
            if params.dataset == 'stl10' and params.more_sample:
                split = 'train+unlabeled'
            else:
                split = 'train'
            dl = get_train_dataset(params.dataset,
                                   method=params.method, split=split)

        elif stage.is_test():
            dl = get_test_dataset(params.dataset)
        else:
            raise NotImplementedError()
        self.regist_dataloader_with_stage(stage, dl)


def main(trainer_cls: ClassVar[Trainer], params_cls: ClassVar[ParamsType]):
    params = params_cls()
    params.from_args()

    dm = SSLDM()
    trainer = trainer_cls(params, dm)

    if params.pretrain_path is not None and params.train_linear:
        trainer.load_state_dict(params.pretrain_path)
        trainer.test()
        return

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_model()
    trainer.save_backbone()
