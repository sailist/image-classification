"""
moco pretrain and its evaluation
"""

from lumo.contrib.scan import base_main, ScanBaseParams

DATE_FLAG = '2023.02.28'


class ScanParams(ScanBaseParams):

    def __init__(self):
        super().__init__()
        self.seeds = 3
        self.dataset = self.choice(*configs.keys())
        self.type = self.choice(*mapping.keys())


def pretrain(pm: ScanParams):
    for dataset in ['cifar10', 'cifar100', 'stl10']:
        if dataset == 'stl10':
            model = 'resnet18'
        else:
            model = 'wrn282'

        yield dict(model=model, config=configs[dataset], scan=f'ssl-{DATE_FLAG}')


mapping = {
    'pretrain': pretrain,
}

configs = {
    'cifar10': 'config/ssl/simclr/cifar10.yaml',
    'cifar100': 'config/ssl/simclr/cifar100.yaml',
    'stl10': 'config/ssl/simclr/stl10.yaml',
}


def main():
    pm = ScanParams()
    pm.from_args()

    files = []
    dics = []
    for kwargs in mapping[pm.type](pm):
        files.append('train_ssl.py')
        dics.append(dict(module='simclr', **kwargs))

    base_main(pm, files, dics)


if __name__ == '__main__':
    main()
