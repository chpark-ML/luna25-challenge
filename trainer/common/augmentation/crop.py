import random


class RandomCrop3DDeprecated:
    def __init__(self, p=0.5, xy_size=72, z_size=48, buffer=8):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.xy_size = xy_size
        self.z_size = z_size
        self.buffer = buffer

    def __call__(self, img, mask=None):
        # assume that the buffer is applied around the given image.
        shift_z, shift_x, shift_y = (self.buffer // 2, self.buffer // 2, self.buffer // 2)
        if random.random() <= self.p:
            # Assume symmetrical voxel size
            shift_z, shift_x, shift_y = (
                (0, 0, 0)
                if self.buffer == 0
                else (
                    random.randrange(self.buffer),
                    random.randrange(self.buffer),
                    random.randrange(self.buffer),
                )
            )
        img = img[
            shift_z : shift_z + self.z_size,
            shift_x : shift_x + self.xy_size,
            shift_y : shift_y + self.xy_size,
        ]

        if mask is not None:
            mask = mask[
                shift_z : shift_z + self.z_size,
                shift_x : shift_x + self.xy_size,
                shift_y : shift_y + self.xy_size,
            ]
            return img, mask
        else:
            return img
