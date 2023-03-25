# cifar10 with wideresnet-28-2

Basic parameters are:

- WideResnet-28-2,
- 200 epochs,
- EMA with 0.999 update ratio,
- SGD(lr=0.1, weight_decay=5e-4, momentum=0.9),
- Cosine decay lr shceduler with 10 linear warmup epochs.

## Results

|  basic methods   | Randaugment  | Simclr's augmentation |
|  --- | --- | --- | 
| 94.81%  | 95.68% | 93.50% |

|  mixup(multihot) | mixup(dual) | Cutmix  | 
|  --- | --- |--- | 
| 95.49% | 95.28%  | 95.62% |

|  mixup(randaug) | mixup(simclr) | Cutmix(randaug)  | Cutmix(simclr) |
|  --- | --- |--- | --- | 
| 94.12% | 93.10%  | 94.82% | 94.01% |

|  BCELoss   | -  | - |
|  --- | --- | --- | 
| -  | - | -|

# cifar100 with resnet18
