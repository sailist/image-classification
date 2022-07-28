"""
半监督长尾分布
"""
import numpy as np
from lumo import DatasetBuilder
from lumo.contrib.data.splits import semi_split

from torch.utils.data import RandomSampler

from augmentations.strategies import (standard_multi_crop, standard, randaugment, simclr, read, basic, none,
                                      simclr_randmask)
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas
from .utils import long_tail_exp_splits


def get_train_dataset(dataset_name, batch_size=64,
                      imb_factor=0.1,
                      batch_count=1024,
                      n_percls=40,
                      mu=7, method='fixmatch',
                      k_size=5, split='train'):
    xs, ys = pick_datas(dataset_name, split=split)

    ys = np.array(ys)
    indexs, img_num_per_cls = long_tail_exp_splits(ys, imb_factor)
    xs = xs[indexs]
    ys = ys[indexs]

    indice_x, indice_un, _ = semi_split(ys, n_percls=n_percls, val_size=0, include_sup=True, repeat_sup=False)
    print(indice_x[:10], indice_un[:10])
    print(len(set(indice_x)), len(set(indice_un)))
    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    sup_ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
            .subset(indice_x)
    )

    if 'stl' in dataset_name and split == 'train':
        uxs, uys = pick_datas(dataset_name, split=split)
        xs = xs + uxs
        ys = ys + np.random.randint(0, 10, len(uxs), dtype=np.int).tolist()

    un_ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('ys', 'ys')
    )
    if 'stl' != dataset_name:
        un_ds.subset(indice_un)

    if lazy_load:
        sup_ds.add_input_transform('xs', read)
        un_ds.add_input_transform('xs', read)

    sup_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))

    if method == 'mixmatch':
        for i in range(k_size):
            un_ds.add_output('xs', f'xs{i}', standard_multi_crop(mean, std, size=img_size,
                                                                 index=i))
    elif method in {'fixmatch', 'flexmatch'}:
        (
            un_ds
                .add_output('xs', 'xs', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs', randaugment(mean, std, size=img_size))
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
                .add_output('xs', 'sxs1', randaugment(mean, std, size=img_size))
        )
    else:  # for default experiments
        un_ds.add_output('xs', 'xs', standard(mean, std, size=img_size))
        (
            sup_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size))
        )
        (
            un_ds
                .add_output('xs', 'xs1', standard(mean, std, size=img_size))
                .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
                .add_output('xs', 'sxs2', randaugment(mean, std, size=img_size))
                .add_output('xs', 'sxs3', randaugment(mean, std, size=img_size))
        )

    ds = (
        DatasetBuilder()
            .add_input('sup', sup_ds.scale_to_size(len(un_ds)))
            .add_input('un', un_ds)
    )
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

    ds.add_output('xs', 'xs', basic(mean, std, size=img_size))
    ds.add_output('xs', 'xsn', none(mean, std, size=img_size))

    return ds
