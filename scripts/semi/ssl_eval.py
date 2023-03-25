"""
Self-supervised learning eval scripts.
"""
from lumo.contrib.scan import base_main, ScanBaseParams

DATE_FLAG = '2023.03.04'


class ScanParams(ScanBaseParams):

    def __init__(self):
        super().__init__()
        self.seeds = 3
        self.dataset = self.choice(*configs.keys())
        self.type = self.choice(*mapping.keys())
        self.pretrain_path = None
        self.model = None


def best(pm: ScanParams):
    """
    Table 1.
    """
    if pm.dataset in {'cifar10', 'cifar100', 'cifar100-big'}:
        for n_percls in [4, 25, 100]:
            yield dict(n_percls=n_percls, scan=f'ssl-semi-eval-{DATE_FLAG}', pretrain_path=pm.pretrain_path,
                       model=pm.model)
    elif pm.dataset == 'stl10':
        yield dict(scan=f'ssl-semi-eval-{DATE_FLAG}', pretrain_path=pm.pretrain_path, model=pm.model)


def baseline(pm: ScanParams):
    """
    Table 1.
    """
    if pm.dataset in {'cifar10', 'cifar100', 'cifar100-big'}:
        for n_percls in [4, 25, 100]:
            yield dict(n_percls=n_percls, scan=f'semi-bsl-{DATE_FLAG}', model=pm.model)
    elif pm.dataset == 'stl10':
        yield dict(scan=f'ssl-semi-eval-{DATE_FLAG}', model=pm.model)


def full(pm: ScanParams):
    for k, v in mapping.items():
        if k == 'full':
            continue
        yield from v(pm)


mapping = {
    'best': best,
    'full': full,
    'baseline': baseline,
}

configs = {
    'cifar10': 'config/semi/ssl_eval/cifar10.yml',
    'cifar100': 'config/semi/ssl_eval/cifar100.yml',
    'cifar100-big': 'config/semi/ssl_eval/cifar100-big.yml',
    'stl10': 'config/semi/ssl_eval/stl10.yml',
}


def main():
    pm = ScanParams()
    pm.from_args()

    files = []
    dics = []
    for kwargs in mapping[pm.type](pm):
        files.append('train_semi.py')
        dics.append(dict(module='crossentropy', config=configs[pm.dataset], **kwargs))

    base_main(pm, files, dics)


if __name__ == '__main__':
    main()
