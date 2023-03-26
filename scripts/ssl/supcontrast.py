"""
moco pretrain and its evaluation
"""

from lumo.contrib.scan import base_main, ScanBaseParams

DATE_FLAG = '2023.02.28'


class ScanParams(ScanBaseParams):

    def __init__(self):
        super().__init__()
        self.seeds = 3
        self.type = self.choice(*mapping.keys())


def pretrain(pm: ScanParams):
    for dataset in ['cifar10', 'cifar100', 'stl10']:
        yield dict(model=pm.get('model', 'resnet18'), scan=f'ssl-{DATE_FLAG}', config=configs[dataset], )


def one(pm: ScanParams):
    yield dict(model=pm.get('model', 'resnet18'), scan=f'ssl-{DATE_FLAG}', config=configs.get('dataset', 'cifar10'), )


mapping = {
    'pretrain': pretrain,
    'one': one,
}

configs = {
    'cifar10': 'config/ssl/supcontrast/cifar10.yaml',
    'cifar100': 'config/ssl/supcontrast/cifar100.yaml',
    'stl10': 'config/ssl/supcontrast/stl10.yaml',
}


def main():
    pm = ScanParams()
    pm.from_args()

    files = []
    dics = []
    for kwargs in mapping[pm.type](pm):
        files.append('train_ssl.py')
        dics.append(dict(module='supcontrast', **kwargs))

    base_main(pm, files, dics)


if __name__ == '__main__':
    main()
