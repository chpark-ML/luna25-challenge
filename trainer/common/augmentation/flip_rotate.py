import random

import numpy as np
from albumentations import RandomRotate90, Rotate


class Flip3D:
    def __init__(self, p=0.5):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, img, mask=None):
        # Apply random flip
        axes = range(1, 8)
        if random.random() <= self.p:
            flipat = np.random.choice(axes)
            flipat = "{0:b}".format(flipat).zfill(3)
            for idx, i in enumerate(flipat):
                if int(i):
                    img = np.flip(img, idx).copy()
                    if mask is not None:
                        mask = np.flip(mask, idx).copy()
        if mask is not None:
            return img, mask
        else:
            return img


class FlipXY:
    def __init__(self, p=0.5):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, img, mask=None):
        if random.random() <= self.p:
            img = img.copy().transpose(0, 2, 1)
            if mask is not None:
                mask = mask.copy().transpose(0, 2, 1)

        if mask is not None:
            return img, mask
        return img


class RandomRotate903D:
    def __init__(self, p=0.5):
        assert 0.0 <= p <= 1.0
        self.rotate = RandomRotate90(p=p)

    def __call__(self, img, mask=None):
        if mask is not None:
            img = np.transpose(img, axes=(1, 2, 0)).copy()
            mask = np.transpose(mask, axes=(1, 2, 0)).copy()
            res = self.rotate(image=img, mask=mask)
            img = res["image"]
            mask = res["mask"]
            img = np.transpose(img, axes=(2, 0, 1))
            mask = np.transpose(mask, axes=(2, 0, 1))
            return img, mask
        else:
            img = np.transpose(img, axes=(1, 2, 0)).copy()
            res = self.rotate(image=img)
            img = res["image"]
            img = np.transpose(img, axes=(2, 0, 1))
            return img
