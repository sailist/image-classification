import numpy as np
import torch
from torch.nn import functional as F
from lumo.contrib.torch.tensor import cartesian_product
from lumo import Meter
from sklearn.mixture import GaussianMixture
from lumo import Logger
from contrib.betamixture import BetaMixture1D

logger = Logger()


def group_fit(features):
    """
    根据特征区分样本是否为噪音标签
    :param features:
    :return:
    """
    # model = GaussianMixture(2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    model = GaussianMixture(2)
    model.fit(features)

    if np.argmax(model.means_[:, 1]) == 0:  # 根据混合分布的均值参数来区分，概率高的将其看做是clean label
        parameters = [i[::-1] for i in model._get_parameters()]
        model._set_parameters(parameters)

    return model


class NLMixture:
    def create_feature(self, *feature):
        feature = np.stack(feature, axis=1)
        return feature

    def gmm_predict(self, feature: np.ndarray, with_prob=True):
        model = group_fit(feature)
        if with_prob:
            res = model.predict_proba(feature)[:, 0]
        else:
            res = model.predict(feature)
        return res

    def norm(self, preds):
        # return preds / np.max(preds)
        return np.clip(preds, 0, 1)

    def bmm_predict(self, feature: np.ndarray, with_prob=True, mean=False, offset=0) -> np.ndarray:
        """
        ress 表示 0 是噪音，1 是干净
        :param feature:
        :param with_prob:
        :param mean:
        :param offset: 将整体的预测概率进行多少的偏移，大于零表示更倾向于保留干净标签，小于零表示更清晰于筛选噪音
        :return:
        """
        _, n = feature.shape
        ress = []
        for i in range(n):
            sub_feature = feature[:, i]
            model = BetaMixture1D()
            model.fit(sub_feature)
            if with_prob:
                res = model.predict_prob(sub_feature)
                nan_mask = np.isnan(res)
                if nan_mask.any():
                    res[nan_mask] = 0
            else:
                res = model.predict(sub_feature)
            ress.append(res)

        if mean:
            ress = sum(ress) / len(ress)
        else:
            ress = sum(ress)

        ress = self.norm(ress + offset)
        logger.info('bmm raw', ress.max(), ress.min())

        return 1 - ress

    def acc_mixture_(self, true_cls: np.ndarray, noisy_cls: np.ndarray, pre='mix'):
        meter = Meter()
        t_n, f_n = '{}t'.format(pre), '{}f'.format(pre)
        meter[t_n] = noisy_cls[true_cls].mean()
        meter[f_n] = noisy_cls[np.logical_not(true_cls)].mean()

        return meter


def euclidean_dist(x, y, min=1e-12):
    """
    copy from https://blog.csdn.net/IT_forlearn/article/details/100022244
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T  # 这里要是 T 不能是 t()，否则会出现莫名的梯度问题
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist = dist - 2 * torch.mm(x, y.T)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=min).sqrt()  # for numerical stability
    return dist


def euclidean_dist_v2(x, y, min=1e-12):
    """
    copy from https://blog.csdn.net/IT_forlearn/article/details/100022244
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    left, right = cartesian_product(x, y)
    dict = F.pairwise_distance(left, right).reshape(m, n)

    return dict


def elementwise_mul(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication of a vector and a matrix

    References:
        https://discuss.pytorch.org/t/element-wise-multiplication-of-a-vector-and-a-matrix/56946

    Args:
        vec: tensor of shape (N, )
        mat: matrix of shape (N, ...)

    Returns:
        A tensor

    """
    nshape = [-1] + [1] * (len(mat.shape) - 1)
    nvec = vec.reshape(nshape)  # .expand_as(mat)
    return nvec * mat
