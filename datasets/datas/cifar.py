from PIL import Image
from lumo.proc.path import cache_dir
from torchvision.datasets import CIFAR10, CIFAR100


def cifar10(split='train'):
    train = split == 'train'
    try:
        dataset = CIFAR10(root=cache_dir(), train=train, download=False)
    except:
        dataset = CIFAR10(root=cache_dir(), train=train, download=True)
    xs = list(Image.fromarray(i) for i in dataset.data)
    ys = list(int(i) for i in dataset.targets)

    return xs, ys


def cifar100(split='train'):
    train = split == 'train'
    try:
        dataset = CIFAR100(root=cache_dir(), train=train, download=False)
    except:
        dataset = CIFAR100(root=cache_dir(), train=train, download=True)
    xs = list(Image.fromarray(i) for i in dataset.data)
    ys = list(int(i) for i in dataset.targets)

    return xs, ys
