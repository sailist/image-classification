"""
不同模块作用对比
"""
import time
from itertools import cycle

import torch
from lumo.utils.memory_grab import memory
from lumo import Params, Logger
from lumo.utils.fmt import strftime
import sys

log = Logger()
log.use_stdout = False
fn = log.add_log_dir(f'./run_scan_methods')


def main():
    pm = Params()

    pm.gpus = torch.cuda.device_count()
    pm.seeds = 3
    pm.enable_gpus = None
    pm.index = 0
    pm.jobs = 1
    pm.from_args()

    log.raw(pm)

    base = (sys.executable +
            " train_nl.py --module=betagl --dataset=cifar10"
            " --c={config} --seed=0 --device={device} --noisy_ratio=0.4 --scan=psensitive-2023.02.25 {extra} & \n")

    sh = []
    c = 0

    extras = [
        '--alpha=0.99',
        '--alpha=0.95',
        '--alpha=0.9',

        '--init_gamma=0.75',
        '--init_gamma=5',
        '--init_gamma=1',

        # '--init_gamma=4',
        # '--init_gamma=2',
        # '--init_gamma=1.25',
        # '--init_gamma=0.5',
        # '--init_gamma=0.25',

        # '--alpha=0.8',
        # '--alpha=0.7',

        # '--ty=0.6',
        # '--ty=0.5',
        # '--ty=0.4',
        # '--ty=0.3',
        # '--ty=0.45',
        # '--ty=0.55',

        '--ty=0.35',
        '--ty=0.65',
        '--ty=0.7',
    ]
    if pm.enable_gpus is None:
        enable_gpus = list(range(pm.gpus))
    else:
        enable_gpus = pm.enable_gpus
        pm.gpus = len(enable_gpus)

    gpus = cycle(enable_gpus)

    for extra in extras:
        device = next(gpus)
        cur = base.format(device=device,
                          extra=extra,
                          config='config/nl/betagl/cifar10.yml')

        # if pm.gpus == 0:[(device + 1) % pm.gpus]
        c = (c + 1) % pm.jobs
        if c % pm.jobs == pm.index:
            memory(3200, device).start()
            print(cur, flush=True)
        time.sleep(20)

    # print(f'echo "execute {len(sh)} tests."')
    # step = pm.gpus if pm.gpus > 0 else 1
    #
    # for i in range(0, len(sh)):
    #     # cmds = sh[i:i + step]
    #     memory(3200, i % pm.gpus).start()
    #     print(sh[i], flush=True)
    #     time.sleep(20)
    print('wait')


if __name__ == '__main__':
    main()
