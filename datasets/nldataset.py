"""
# Reference
https://github.com/libffcv/ffcv-imagenet
https://arxiv.org/abs/1906.06423
"""
from lumo import DatasetBuilder, Logger
from lumo.contrib.data.splits import train_val_split

from augmentations.strategies import standard, simclr, read, randaugment, basic, none
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas
import numpy as np


def asymmetric_noisy(train_y, noisy_ratio, n_classes=None) -> np.ndarray:
    if n_classes is None:
        n_classes = len(set(train_y))
    noisy_map = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}

    noisy_ids = np.random.permutation(len(train_y))[:int(noisy_ratio * len(train_y))]
    noisy_ids = set(noisy_ids)
    noisy_y = np.array(train_y)

    cls_lis = []

    for i in range(n_classes):
        cls_lis.append(list(set(np.where(noisy_y == i)[0]) & noisy_ids))

    for i, idq in enumerate(cls_lis):
        if i in noisy_map:
            noisy_y[idq] = noisy_map[i]

    return noisy_y


def symmetric_noisy(train_y: np.ndarray, noisy_ratio: float, n_classes: int = None, force=False) -> np.ndarray:
    """

    :param train_y: raw clean labels
    :param noisy_ratio:
    :param n_classes:
    :param force:
    :return:
    """
    if n_classes is None:
        n_classes = len(set(train_y))

    noisy_ids = np.random.permutation(len(train_y))[:int(noisy_ratio * len(train_y))]
    noisy_y = np.array(train_y)

    _noisys = np.random.randint(0, n_classes, noisy_ids.shape[0])

    if force:
        _mask = np.where(_noisys == noisy_y[noisy_ids])
        while _mask[0].shape[0] != 0:
            _noisys[_mask] = np.random.randint(0, 10, _mask[0].shape[0])
            _mask = np.where(_noisys == noisy_y[noisy_ids])

    noisy_y[noisy_ids] = _noisys

    return noisy_y


def get_train_dataset(dataset_name, method='default', split='train', noisy_ratio=0.8, noisy_type='asymmetric'):
    xs, ys = pick_datas(dataset_name, split=split)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = reimg_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds
    n_classes = len(set(ys))

    if noisy_type == 'asymmetric':
        noisy_ys = asymmetric_noisy(ys, noisy_ratio, n_classes=n_classes)
    elif noisy_type == 'symmetric':
        noisy_ys = symmetric_noisy(ys, noisy_ratio, n_classes=n_classes)
    else:
        raise NotImplementedError()

    ds = (
        DatasetBuilder()
            .add_idx('ids')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_input('nys', noisy_ys)
            .add_output('xs', 'xs', none(mean, std, size=img_size))
            .add_output('xs', 'weak', standard(mean, std, size=img_size, resize=reimg_size))
            .add_output('xs', 'strong', randaugment(mean, std, size=img_size, resize=reimg_size))
            .add_output('xs', 'simclr', simclr(mean, std, size=img_size, resize=reimg_size))
            .add_output('ys', 'tys')
            .add_output('ys', 'ys')
            .add_output('nys', 'nys')
    )
    Logger().info(f'Label accuracy with {noisy_type}/{noisy_ratio}: {(np.array(noisy_ys) == np.array(ys)).mean()}')

    if lazy_load:
        ds.add_input_transform('xs', read)

    return ds


def get_test_dataset(dataset_name):
    xs, ys = pick_datas(dataset_name, split='test')

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    # ds.add_output('xs', 'xs', basic(mean, std, size=img_size, resize=img_size))
    ds.add_output('xs', 'xs', none(mean, std, size=img_size))

    return ds


from torchvision.datasets import ImageFolder
