"""
基线对比
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
    pm.enable_gpus = None
    pm.seeds = 3
    pm.index = 0
    pm.jobs = 1
    pm.from_args()

    log.raw(pm)

    base = (sys.executable +
            " train_nl.py --module=betagl --dataset=cifar10"
            " --c={config} --seed={seed} --device={device} --noisy_ratio={noisy_ratio} --scan=2023.02.27 & \n")

    device = 0

    if pm.enable_gpus is None:
        enable_gpus = list(range(pm.gpus))
    else:
        enable_gpus = pm.enable_gpus
        pm.gpus = len(enable_gpus)

    gpus = cycle(enable_gpus)

    sh = []
    c = 0
    for seed in range(pm.seeds):
        for noisy_ratio in [0.4, 0.6]:
            device = next(gpus)
            cur = base.format(seed=seed,
                              device=device,
                              noisy_ratio=noisy_ratio,
                              config='config/nl/betagl/cifar10.yml')

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
