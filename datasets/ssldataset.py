from lumo import DatasetBuilder

from augmentations.strategies import (standard, simclr, read, basic, none, rotate,
                                      standard_rotate, standard_resize)
from .const import mean_std_dic, imgsize_dic, lazy_load_ds
from .datas import pick_datas


def get_train_dataset(dataset_name, method='basic', split='train'):
    xs, ys = pick_datas(dataset_name, split=split)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = DatasetBuilder()
    (ds
     .add_idx('id')
     .add_input('xs', xs)
     .add_input('ys', ys))
    if method == 'essl':
        ds.add_output('xs', 'xsu', rotate(mean, std, size=img_size // 2, v=0))
        ds.add_output('xs', 'xsd', rotate(mean, std, size=img_size // 2, v=180))
        ds.add_output('xs', 'xsl', rotate(mean, std, size=img_size // 2, v=90))
        ds.add_output('xs', 'xsr', rotate(mean, std, size=img_size // 2, v=-90))
    else:
        (ds
         .add_output('xs', 'xs', none(mean, std, size=img_size))
         .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
         .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
         .add_output('ys', 'ys')
         )
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

    ds.add_output('xs', 'xs0', basic(mean, std, size=img_size))
    ds.add_output('xs', 'xs1', none(mean, std, size=img_size))

    return ds
