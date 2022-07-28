mean_std_dic = {
    'cifar10': [(0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)],
    'cifar100': [(0.5071, 0.4867, 0.4408),
                 (0.2675, 0.2565, 0.2761)],
    'svhn': [(0.44921358705286946, 0.4496640988868895, 0.45029627318846444),
             (0.2003216966442779, 0.1991626631851053, 0.19936594996908613)],
    'stl10': [(0.44319644335512015, 0.4463139215686274, 0.44558495098039186),
              (0.26640707592603363, 0.2644222394907146, 0.2637064714107059)],
    'default': [(0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)],

}

# from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/ResNet18_224.ipynb
for k in ['tinyimagenet', 'tinyimagenet-64']:
    mean_std_dic[k] = [(0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)]

# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
for k in ['imagenet', 'imagenet-64', 'imagenet-96', 'imagenet-112']:
    mean = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

imgsize_dic_ = {
    'cifar10': 32,
    'cifar10-64': 64,
    'cifar10-96': 96,
    'cifar100': 32,
    'cifar100-64': 32,
    'cifar100-96': 32,
    'stl10': 96,
    'svhn': 32,
    'tinyimagenet': 72,
    'tinyimagenet-64': 64,
    'tinyimagenet-96': 96,
    'tinyimagenet-112': 112,
    'tinyimagenet-224': 224,
    'imagenet': 224,
    'imagenet-64': 64,
    'imagenet-96': 96,
    'imagenet-112': 112,
}


def imgsize_dic(dataset_name):
    res = dataset_name.split('-')
    assert res[0] in imgsize_dic_
    if len(res) == 2:
        return int(res[1])
    return imgsize_dic_[res[0]]


lazy_load_ds = {
    'tinyimagenet',
    'tinyimagenet-64',
    'tinyimagenet-96',
    'tinyimagenet-112',
    'tinyimagenet-224',
    'imagenet',
    'imagenet-64',
    'imagenet-96',
    'imagenet-112',
    'clothing1m'
}

n_classes = {
    'cifar10': 10,
    'cifar10-64': 10,
    'cifar10-96': 10,
    'cifar100': 100,
    'cifar100-64': 100,
    'cifar100-96': 100,
    'svhn': 10,
    'stl10': 10,
    'tinyimagenet': 200,
    'tinyimagenet-64': 200,
    'tinyimagenet-96': 200,
    'tinyimagenet-112': 200,
    'tinyimagenet-224': 200,
    'imagenet': 1000,
    'imagenet-64': 1000,
    'imagenet-96': 1000,
    'imagenet-112': 1000,
}

cifar100_coarse_label_map = {
    19: 11, 29: 15, 0: 4, 11: 14, 1: 1, 86: 5, 90: 18, 28: 3, 23: 10, 31: 11, 39: 5,
    96: 17, 82: 2, 17: 9, 71: 10, 8: 18, 97: 8, 80: 16, 74: 16, 59: 17, 70: 2, 87: 5,
    84: 6, 64: 12, 52: 17, 42: 8, 47: 17, 65: 16, 21: 11, 22: 5, 81: 19, 24: 7, 78: 15,
    45: 13, 49: 10, 56: 17, 76: 9, 89: 19, 73: 1, 14: 7, 9: 3, 6: 7, 20: 6, 98: 14,
    36: 16, 55: 0, 72: 0, 43: 8, 51: 4, 35: 14, 83: 4, 33: 10, 27: 15, 53: 4, 92: 2,
    50: 16, 15: 11, 18: 7, 46: 14, 75: 12, 38: 11, 66: 12, 77: 13, 69: 19, 95: 0, 99: 13,
    93: 15, 4: 0, 61: 3, 94: 6, 68: 9, 34: 12, 32: 1, 88: 8, 67: 1, 30: 0, 62: 2, 63: 12,
    40: 5, 26: 13, 48: 18, 79: 13, 85: 19, 54: 2, 44: 15, 7: 7, 12: 9, 2: 14, 41: 19,
    37: 9, 13: 18, 25: 6, 10: 3, 57: 4, 5: 6, 60: 10, 91: 1, 3: 8, 58: 18, 16: 3,
}
