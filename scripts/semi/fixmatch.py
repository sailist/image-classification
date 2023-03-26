"""
CHMatch reimplement scripts.
"""
from lumo.contrib.scan import base_main, ScanBaseParams

DATE_FLAG = '2023.03.04'


class ScanParams(ScanBaseParams):

    def __init__(self):
        super().__init__()
        self.seeds = 3
        self.dataset = self.choice(*configs.keys())
        self.type = self.choice(*mapping.keys())


def best(pm: ScanParams):
    if pm.dataset in {'cifar10', 'cifar100', 'cifar100-big'}:
        for seed in range(pm.seeds):
            for n_percls in [4, 25, 100]:
                yield dict(n_percls=n_percls, scan=f'fixmatch-{DATE_FLAG}', seed=seed)
    elif pm.dataset == 'stl10':
        for seed in range(pm.seeds):
            yield dict(scan=f'fixmatch-{DATE_FLAG}', seed=seed, n_percls=100)


mapping = {
    'best': best,
}

configs = {
    'cifar10': 'config/semi/fixmatch/cifar10.yml',
    'cifar100': 'config/semi/fixmatch/cifar100.yml',
    'cifar100-big': 'config/semi/fixmatch/cifar100-big.yml',
    'stl10': 'config/semi/fixmatch/stl10.yml',
}


def main():
    pm = ScanParams()
    pm.from_args()

    files = []
    dics = []
    for kwargs in mapping[pm.type](pm):
        files.append('train_semi.py')
        dics.append(dict(module='fixmatch', config=configs[pm.dataset], **kwargs))
    base_main(pm, files, dics)


if __name__ == '__main__':
    main()
