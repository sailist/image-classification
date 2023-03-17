"""
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)


num_class: 16185
train: 8144
test: 8041


```
mkdir cardatasets
cd cardatasets
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat


```

"""

import scipy.io

from pathlib import Path


def main(root):
    mat = scipy.io.loadmat('cardatasets/devkit/cars_meta.mat')
