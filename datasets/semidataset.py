"""
半监督数据集
"""
import numpy as np
from lumo import DatasetBuilder
from lumo.contrib.data.splits import semi_split
from torch.utils.data import RandomSampler

from augmentations.strategies import (standard_multi_crop, standard, randaugment, simclr, read, basic, none,
                                      simclr_randmask)
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas
from lumo import Logger

log = Logger()


def get_train_dataset(dataset_name, n_percls=40,
                      method='fixmatch',
                      k_size=5,
                      split='train'):
    xs, ys = pick_datas(dataset_name, split=split)

    log.raw('Load train dataset:', len(xs))
    indice_x, indice_un, _ = semi_split(ys, n_percls=n_percls, val_size=0, include_sup=True, repeat_sup=False)
    log.raw('Seed check: ', indice_x[:10], indice_un[:10])
    log.raw('Split into sup/unsup:', len(set(indice_x)), len(set(indice_un)))
    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    # if 'stl' in dataset_name and split == 'train':
    #     uxs, uys = pick_datas(dataset_name, split='unlabeled')
    #     xs = xs + uxs
    #     ys = ys + np.random.randint(0, 9, len(uxs), dtype=np.int).tolist()
    #     print('xs', len(xs))

    sup_ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
            .subset(indice_x)
    )
    un_ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
    )

    nm = (3, 5)
    if 'stl' not in dataset_name:
        un_ds.subset(indice_un)
        nm = (2, 10)

    if lazy_load:
        sup_ds.add_input_transform('xs', read)
        un_ds.add_input_transform('xs', read)

    sup_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))

    if method == 'baseline':
        (
            un_ds
                .add_idx('ids')
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
        )
    elif method == 'mixmatch':
        for i in range(k_size):
            un_ds.add_output('xs', f'xs{i}', standard_multi_crop(mean, std, size=img_size,
                                                                 index=i))
    elif method in {'fixmatch', 'flexmatch'}:
        (
            un_ds
                .add_idx('ids')
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs', randaugment(mean, std, size=img_size, nm=nm))
        )
    elif method == 'comatch':
        (
            sup_ds
                .add_output('xs', 'sxs0', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
        )
        (
            un_ds
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', randaugment(mean, std, size=img_size, nm=nm))
        )
    elif method == 'chmatch':
        (sup_ds.add_output('xs', 'xs1', standard(mean, std, size=img_size)))
        (
            un_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'simclr', simclr(mean, std, size=img_size))
                .add_output('xs', 'randaug', randaugment(mean, std, size=img_size, nm=nm))
        )

    else:  # for default experiments
        un_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))
        (
            sup_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size, nm=nm))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size, nm=nm))
        )
        (
            un_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size, nm=nm))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size, nm=nm))
        )

    return sup_ds, un_ds


def get_test_dataset(dataset_name):
    xs, ys = pick_datas(dataset_name, split='test')

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None
    log.raw('Load test dataset:', len(xs))
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

    ds.add_output('xs', 'xs', basic(mean, std, size=img_size))
    return ds


from torchvision import datasets
