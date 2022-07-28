from PIL import Image
from lumo.proc.path import cache_dir
from torchvision.datasets import SVHN


def svhn(split='train'):
    """
    32x32x3
    """
    try:
        dataset = SVHN(root=cache_dir(), split=split, download=False)
    except:
        dataset = SVHN(root=cache_dir(), split=split, download=True)
    xs = list(Image.fromarray(i) for i in dataset.data)
    ys = list(int(i) for i in dataset.labels)

    return xs, ys


def svhn_extra(split='train'):
    """
    32x32x3
    """
    try:
        dataset = SVHN(root=cache_dir(), split=split, download=False)
    except:
        dataset = SVHN(root=cache_dir(), split=split, download=True)
    xs = list(Image.fromarray(i) for i in dataset.data)
    ys = list(int(i) for i in dataset.labels)

    return xs, ys
