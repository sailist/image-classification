"""
基线对比
"""
import time
from itertools import cycle

import torch
from lumo.utils.memory_grab import memory
from lumo import Params, Logger
import sys

log = Logger()
log.use_stdout = False
fn = log.add_log_dir(f'./run_scan_methods')

DATE_FLAG = '2023.02.27'


def format_args(**kwargs):
    return ' '.join([f'--{k}=={v}' for k, v in kwargs.items()])


def best():
    for noisy_ratio in [0.2, 0.4, 0.6, 0.8]:
        for seed in range(3):
            yield format_args(noisy_ratio=noisy_ratio, scan=f'best-{DATE_FLAG}', seed=seed)


def psensitive():
    scan = f'psens-{DATE_FLAG}'
    noisy_ratio = 0.4
    for alpha in [0.7, 0.8, 0.9, 0.95, 0.99]:
        yield format_args(alpha=alpha, scan=scan, noisy_ratio=noisy_ratio)

    for init_gamma in [
        0.25, 0.5, 0.75, 1.25,
        1, 2, 4, 5,
    ]:
        yield format_args(init_gamma=init_gamma, scan=scan, noisy_ratio=noisy_ratio)

    for ty in [
        0.6, 0.5, 0.4,
        0.3, 0.7,
        0.35, 0.45, 0.55, 0.65,
    ]:
        yield format_args(ty=ty, scan=scan, noisy_ratio=noisy_ratio)


def module_ab():
    noisy_ratio = 0.4
    for k, v in [
        ('local_filter', False),
        ('init_gamma', 0),
        ('global_filter', False),
        ('global_mixture', 'gmm'),
    ]:
        yield format_args(**{k: v}, scan=f'm-ab-{DATE_FLAG}', noisy_ratio=noisy_ratio)


def full():
    for k, v in mapping.items():
        if k == 'full':
            continue
        yield from v()


mapping = {
    'ab': module_ab,
    'sens': psensitive,
    'best': best,
    'full': full,
}

configs = {
    'cifar10': 'config/nl/betagl/cifar10.yml',
    'cifar100': 'config/nl/betagl/cifar100.yml',
}


def main():
    pm = Params()

    pm.gpus = None
    pm.seeds = 3
    pm.dataset = pm.choice(*configs.keys())
    pm.type = pm.choice(*mapping.keys())
    pm.from_args()

    log.raw(pm)

    base = (sys.executable +
            " train_nl.py --module=betagl --c={config} --device={device} --extra={extra} & \n")

    if not torch.cuda.is_available():
        gpu_lis = ['cpu']
    elif pm.gpus is None:
        gpu_lis = list(range(torch.cuda.is_available()))
    elif isinstance(pm.gpus, (int, str)):
        gpu_lis = [torch.device(pm.gpus).index]
    else:
        gpu_lis = pm.gpus

    gpu_lis = cycle(gpu_lis)

    for extra in mapping[pm.type]():
        device = next(gpu_lis)
        cur = base.format(extra=extra,
                          device=device,
                          config=configs[pm.dataset])
        memory(3200, device).start()
        log.info(cur.strip())
        print(cur, flush=True)
        time.sleep(20)

    print('wait')


if __name__ == '__main__':
    main()
