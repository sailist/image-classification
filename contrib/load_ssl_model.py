from typing import Any

import torch
from lumo import Trainer, ParamsType
from lumo.trainer import callbacks


class SSLLoadModel(callbacks.InitialCallback):
    """
    Callback to automatically load pretrained model.

    The load function will be executed if and only if
     - both the params `pretrain` and `pretrain_path` are defined in ParamsType
     - `pretrain` is True and `pretrain_path` is not None.
    """

    def on_imodels_end(self, trainer: Trainer, func, params: ParamsType,
                       result: Any, *args, **kwargs):
        super().on_imodels_end(trainer, func, params, result, *args, **kwargs)
        if params.get("pretrain", False):
            path = params.get("pretrain_path", None)
            if path is not None:
                state_dict = torch.load(path, map_location="cpu")
                trainer.logger.info(f'load pretrain parameters from {path}')
                model_state_dict = {
                    k.replace("backbone.", ""): v
                    for k, v in state_dict["models"]["model"].items()
                    if "backbone" in k
                }
                ema_state_dict = {
                    k.replace("backbone.", ""): v
                    for k, v in state_dict["models"]["ema_model"].items()
                    if "backbone" in k
                }

                trainer.logger.info(f'load pretrain paramteres to trainer.model.backbone: '
                                    f'{trainer.model.backbone.load_state_dict(model_state_dict)}')
                if params.get('ema', False) and getattr(trainer, 'ema_model', None) is not None:
                    trainer.logger.info(f'load pretrain paramteres to trainer.ema_model.backbone: '
                                        f'{trainer.ema_model.backbone.load_state_dict(model_state_dict)}')
