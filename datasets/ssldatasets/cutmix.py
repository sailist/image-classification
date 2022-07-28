from augmentations.strategies import standard, simclr, read

from ..datas import regist_data
from ..const import mean_std_dic, imgsize_dic, lazy_load_ds
from lumo import DatasetBuilder


def get_train_dataset(dataset_name, method='basic'):
    data_fn = regist_data.get(dataset_name, None)
    assert data_fn is not None
    xs, ys = data_fn(train=True)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
            .add_idx('id')
            .add_input('xs', xs)
            .add_input('ys', ys)
            .add_output('xs', 'xs', standard(mean, std, size=img_size))
            .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
            .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
            .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    return ds


def get_test_dataset(dataset_name):
    data_fn = regist_data.get(dataset_name, None)
    assert data_fn is not None
    xs, ys = data_fn(train=False)

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

    ds.add_output('xs', 'xs', standard(mean, std, size=img_size))

    return ds
