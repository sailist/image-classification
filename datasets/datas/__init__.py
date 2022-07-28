from . import cifar, clothing1m, svhn, stl10, imagenet

regist_data = {
    'cifar10': cifar.cifar10,
    'cifar100': cifar.cifar100,
    'clothing1m': None,
    'svhn': svhn.svhn,
    'stl10': stl10.stl10,
    'tinyimagenet': imagenet.tiny_imagenet200,
    'imagenet': imagenet.imagenet,
}


def pick_datas(dataset_name: str, split='train'):
    dataset_name = dataset_name.split('-')[0]
    data_fn = regist_data.get(dataset_name, None)
    assert data_fn is not None
    xs, ys = data_fn(split=split)
    return xs, ys
