"""
基线对比
"""
import time

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
    pm.index = 0
    pm.jobs = 1
    pm.from_args()

    log.raw(pm)

    base = (sys.executable +
            " train_nl.py --module=betagl --dataset=cifar10"
            " --c={config} --seed={seed} --device={device} --noisy_ratio={noisy_ratio} --scan=2023.02.16 & \n")

    device = 0

    sh = []
    c = 0
    for seed in range(pm.seeds):
        for noisy_ratio in [0.2, 0.4, 0.6, 0.8]:
            cur = base.format(seed=seed,
                              device=device,
                              noisy_ratio=noisy_ratio,
                              config='config/nl/betagl/cifar100.yml')

            if pm.gpus == 0:
                device = 'cpu'
            else:
                device = (device + 1) % pm.gpus
            c = (c + 1) % pm.jobs
            if c % pm.jobs == pm.index:
                sh.append(cur)

    print(f'echo "execute {len(sh)} tests."')
    step = pm.gpus if pm.gpus > 0 else 1

    for i in range(0, len(sh)):
        # cmds = sh[i:i + step]
        memory(3200, i % pm.gpus).start()
        print(sh[i], flush=True)
        time.sleep(20)
    print('wait')


if __name__ == '__main__':
    main()
