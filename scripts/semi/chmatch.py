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


def ab_k_increasing(pm: ScanParams):
    """
    K increasing manner: linear/exponential/log
    Supplementary
    """
    for n_percls in [4, 25, 100]:
        for k_type in ['linear', 'cos', 'log', 'exp']:
            yield dict(k_type=k_type, n_percls=n_percls, scan=f'ab-k-inc-{DATE_FLAG}')


def ab_module(pm: ScanParams):
    """
    Ablation study of Graph matching and Threshold strategies.

    Add --dataset=cifar100 for reproducing Table 4. results.
    """
    # use fixmatch threshold
    # line 1
    yield dict(proportion_thresh=False, n_percls=4, scan=f'ab-module-{DATE_FLAG}')
    # line 2
    yield dict(k_type='fixed', n_percls=4, scan=f'ab-module-{DATE_FLAG}')
    # line 3
    yield dict(apply_h_mask=False, n_percls=4, scan=f'ab-module-{DATE_FLAG}')
    # line 4
    # ? not sure is this argument for line 4
    yield dict(include_ssl=False, n_percls=4, scan=f'ab-module-{DATE_FLAG}')
    # line 5 result if from the best


def ab_loss_weight(pm: ScanParams):
    """
    Ablation study of alpha, beta, and gamma
    Table 5.
    """
    for name in ['w_alpha', 'w_alpha', 'w_gamma']:
        for w in [0.75, 1, 1.5, 2]:
            yield dict(n_percls=4, **{name: w}, scan=f'ab-lw-{DATE_FLAG}')


def best(pm: ScanParams):
    """
    Table 1.
    """
    if pm.dataset in {'cifar10', 'cifar100', 'cifar100-big'}:
        for seed in range(pm.seeds):
            for n_percls in [4, 25, 100]:
                yield dict(n_percls=n_percls, scan=f'best-{DATE_FLAG}', seed=seed)
    elif pm.dataset == 'stl10':
        for seed in range(pm.seeds):
            yield dict(scan=f'best-{DATE_FLAG}', seed=seed, n_percls=100)


def ab_k_epoch(pm: ScanParams):
    """
    Influence of dynamic duration and max value.
    Fig. 3(a) and (b)
    """
    # 3(a)
    for k_inc_epoch in [50, 75, 100, 125, 150]:
        yield dict(k_inc_epoch=k_inc_epoch)
    # 3(b)
    for max_proportion in [0.70, 0.75, 0.80, 0.85, 0.90]:
        yield dict(max_proportion=max_proportion)


def full(pm: ScanParams):
    for k, v in mapping.items():
        if k == 'full':
            continue
        yield from v(pm)


mapping = {
    'best': best,
    'full': full,
    'ab_k_increasing': ab_k_increasing,
    'ab_module': ab_module,
    'ab_loss_weight': ab_loss_weight,
    'ab_k_epoch': ab_k_epoch,
}

configs = {
    'cifar10': 'config/semi/chmatch/cifar10.yml',
    'cifar100': 'config/semi/chmatch/cifar100.yml',
    'cifar100-big': 'config/semi/chmatch/cifar100-big.yml',
    'stl10': 'config/semi/chmatch/stl10.yml',
}


def main():
    pm = ScanParams()
    pm.from_args()

    files = []
    dics = []
    for kwargs in mapping[pm.type](pm):
        files.append('train_semi.py')
        dics.append(dict(module='chmatch', config=configs[pm.dataset], **kwargs))
    base_main(pm, files, dics)


if __name__ == '__main__':
    main()
