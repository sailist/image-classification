from lumo import DatasetBuilder
from lumo.utils.random import set_state, get_state

from augmentations.strategies import standard, randaugment, simclr, read, basic, none
from torchvision.transforms import transforms
from augmentations.components.label_augment import RandAugmentPC
from ..const import mean_std_dic, imgsize_dic, lazy_load_ds
from ..datas import regist_data
from lumo.data.collate import CollateBase


class TransformLabeler(CollateBase):

    def __init__(self, dataset_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
        img_size = imgsize_dic(dataset_name)
        self.simclr = simclr(mean, std, size=img_size)
        self.rand = RandAugmentPC(1, 10)
        self.totensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

    def before_collate(self, sample_list):
        batch_size = len(sample_list)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            state = get_state(cuda=False)
            sa, sb = sample_list[i], sample_list[j]

            set_state(state, cuda=False)
            sa['xs1'], sa['tys1'] = self.rand(sa['xs'])
            sa['xs'], sa['tys'] = self.rand(sa['xs'])
            set_state(state, cuda=False)
            sb['xs1'], sb['tys1'] = self.rand(sb['xs'])
            sb['xs'], sb['tys'] = self.rand(sb['xs'])

        for i in range(batch_size):
            sample_list[i]['xs1'] = self.totensor(sample_list[i]['xs1'])
            sample_list[i]['xs'] = self.totensor(sample_list[i]['xs'])

        return super().before_collate(sample_list)


def get_train_dataset(dataset_name):
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
            .add_output('ys', 'ys')
            .add_output('xs', 'xs')
            .add_output('xs', 'wxs0', simclr(mean, std, size=img_size))
            .add_output('xs', 'wxs1', simclr(mean, std, size=img_size))
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

    ds.add_output('xs', 'xs0', basic(mean, std, size=img_size))
    ds.add_output('xs', 'xs1', none(mean, std, size=img_size))

    return ds
