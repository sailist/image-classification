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
        for seed in range(0):
            yield format_args(noisy_ratio=noisy_ratio, scan=f'best-{DATE_FLAG}', seed=seed)


def full():
    for k, v in mapping.items():
        if k == 'full':
            continue
        yield from v()


mapping = {
    'best': best,
    'full': full,
}

configs = {
    'cifar10': 'config/nl/crossentropy/cifar10.yml',
    'cifar100': 'config/nl/crossentropy/cifar100.yml',
}


def main():
    pm = Params()

    pm.gpus = None
    pm.seeds = 3
    pm.dataset = pm.choice('cifar10', 'cifar100')
    pm.type = pm.choice(*mapping.keys())
    pm.from_args()

    log.raw(pm)

    base = (sys.executable +
            " train_nl.py --module=crossentropy --dataset={dataset}"
            " --c={config} --device={device} --extra={extra} & \n")

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
                          dataset=pm.dataset,
                          config=configs[pm.dataset])
        memory(3200, device).start()
        print(cur, flush=True)
        time.sleep(20)

    print('wait')


if __name__ == '__main__':
    main()
