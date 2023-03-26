import numpy as np
from lumo.core.interp import Cos


class FixMatchLrSche(Cos):

    def __call__(self, cur):
        ratio = np.cos((7 * np.pi * cur) / (16 * self.right))
        return self.start * ratio
