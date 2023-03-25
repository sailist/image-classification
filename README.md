# image-classification

Reimplemented papers of supervised/semi-supervised/self-supervised learning


# Reimplemented methods

> (done flag means the results are validated)

- Self-supervised-learning , see [ssl-knn.csv](./results/ssl-knn.csv) for knn results.
    - [x] SimCLR V2
    - [x] MoCo V2
    - [ ] SupContrast

- Semi-supvised Learning, see [semi-supervised-learning.csv](./results/semi-supervised-learning.csv).
    - [x] CHMatch
    - [ ] FlexMatch
    - [ ] FixMatch
    - [ ] CoMatch
    - [ ] MixMatch
- Supvised-learning


# How to reproduce

Please refer to the issue links corresponding to the `meta_info` column in the csv files.

# Add your method

You may need to prepare: dataset (most of them are ready), data augmentation methods (most cases are ready), model (most of them are ready), and training code (core logic part).

## Dataset

The method for reading all basic data sources is defined in [datasets/datas/](./datasets/datas/init.py), which provides (xs, ys) sample pairs for most data. If the format you need is not in this directory, you need to implement it yourself.

## Transform
Common augmentation methods are defined in [augmentations/strategies.py](./augmentations/strategies.py). If what you need is not available, you can add one in the same format.

To use these augmentation strategies, taking self-supervised learning as an example, two methods are defined in，[在dataset/ssldataset.py](./datasets/ssldataset.py): 

```python
def get_train_dataset(dataset_name, method='basic', split='train'): ...
def get_test_dataset(dataset_name): ...
```

You can change the augmentation method you need in the method:

```python
if method == 'essl':
    ds.add_output('xs', 'xsu', rotate(mean, std, size=img_size // 2, v=0))
    ds.add_output('xs', 'xsd', rotate(mean, std, size=img_size // 2, v=180))
    ds.add_output('xs', 'xsl', rotate(mean, std, size=img_size // 2, v=90))
    ds.add_output('xs', 'xsr', rotate(mean, std, size=img_size // 2, v=-90))
elif method == 'you method':
    ...
else:
    (ds
     .add_output('xs', 'xs', none(mean, std, size=img_size))
     .add_output('xs', 'sxs0', simclr(mean, std, size=img_size))
     .add_output('xs', 'sxs1', simclr(mean, std, size=img_size))
     .add_output('ys', 'ys')
     )
```

## 模型 & 训练

Taking self-supervised learning as an example, you can copy code such as [simclr](./track_ssl/simclr.py)/[moco](./track_ssl/moco.py), and implement your own code based on it. In most cases, the test code of the same track does not need to be rewritten.

> The method is determined by Params.method, and the module is determined by the part of your training file without the suffix (os.path.basename()).

