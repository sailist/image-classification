"""
不同模块作用对比
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
            " --c={config} --seed=0 --device={device} --noisy_ratio=0.4 --scan=module-ab-2023.02.25 {extra} & \n")

    device = 0

    sh = []
    c = 0

    extras = [
        '--local_filter=False',
        '--init_gamma=0',
        '--global_filter=False',
        '--global_mixture=gmm',
    ]

    for extra in extras:
        cur = base.format(device=device,
                          extra=extra,
                          config='config/nl/betagl/cifar10.yml')

        if pm.gpus == 0:
            device = 'cpu'
        else:
            device = (device + 1) % pm.gpus
        c = (c + 1) % pm.jobs
        if c % pm.jobs == pm.index:
            sh.append(cur)

    print(f'echo "execute {len(sh)} tests."')

    for i in range(0, len(sh)):
        # cmds = sh[i:i + step]
        memory(3200, i % pm.gpus).start()
        print(sh[i], flush=True)
        time.sleep(20)
    print('wait')


if __name__ == '__main__':
    main()
