from torch.utils.data.dataloader import DataLoader
from lumo.core import BaseParams
from . import const


class DataloaderParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.batch_size = 100
        self.pin_memory = True
        self.num_workers = 8


class DataParams(BaseParams):

    def __init__(self):
        super().__init__()
        self.dataset = self.choice(
            *list(const.n_classes.keys())
        )
        self.n_classes = 10

        self.train = DataloaderParams().from_kwargs(shuffle=True)
        self.test = DataloaderParams().from_kwargs(shuffle=False)

    def iparams(self):
        super().iparams()
        self.n_classes = const.n_classes[self.dataset]
