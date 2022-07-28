"""
# Exploring Simple Siamese Representation Learning
 - [paper](https://arxiv.org/abs/2011.10566)
 - [code](https://github.com/lucidrains/byol-pytorch)
"""
from .byol import *


class SimSiamParams(BYOLParams):

    def __init__(self):
        super().__init__()
        self.apply_simsiam = True


ParamsType = SimSiamParams


class SimSiamTrainer(BYOLTrainer):
    pass


TrainerType = SimSiamTrainer

main = partial(main, SimSiamTrainer, SimSiamTrainer)
