import random

import numpy as np


class CoarseDropout3D:
    def __init__(
            self,
            p=0.5,
            patch_size=(72, 72, 72),
            max_holes=12,
            size_limit=(0.05, 0.1),
            overlap_thr=0.25,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.patch_size = patch_size
        self.max_holes = max_holes
        self.size_limit = size_limit
        self.overlap_thr = overlap_thr

    def __call__(self, img, mask=None):
        if random.random() <= self.p:
            if mask is not None:
                img = np.array(img, copy=True)
                mask = np.array(mask, copy=True)
                volume = np.sum(mask >= 0.5)
                for i in range(np.random.randint(self.max_holes)):
                    _volume = np.sum(mask >= 0.5)
                    if _volume / volume <= self.overlap_thr * 2:  # drop threshold in total
                        break

                    box_size = np.random.uniform(*self.size_limit) * np.min(self.patch_size)
                    topleft = np.array([np.random.randint(0, self.patch_size[j] - box_size) for j in range(3)])
                    bottomright = (topleft + box_size).astype(int)

                    inter = np.sum(
                        mask[
                        topleft[0]: bottomright[0],
                        topleft[1]: bottomright[1],
                        topleft[2]: bottomright[2],
                        ]
                    )
                    if inter / box_size ** 3 > self.overlap_thr:  # drop threshold at once
                        continue
                    else:
                        img[
                        topleft[0]: bottomright[0],
                        topleft[1]: bottomright[1],
                        topleft[2]: bottomright[2],
                        ] = 0.0
                        mask[
                        topleft[0]: bottomright[0],
                        topleft[1]: bottomright[1],
                        topleft[2]: bottomright[2],
                        ] = 0.0
            else:
                img = np.array(img, copy=True)
                for i in range(np.random.randint(self.max_holes)):
                    box_size = np.random.uniform(*self.size_limit) * np.min(self.patch_size)
                    topleft = np.array([np.random.randint(0, self.patch_size[j] - box_size) for j in range(3)])
                    bottomright = (topleft + box_size).astype(int)
                    img[
                    topleft[0]: bottomright[0],
                    topleft[1]: bottomright[1],
                    topleft[2]: bottomright[2],
                    ] = 0.0
        if mask is not None:
            return img, mask

        return img
