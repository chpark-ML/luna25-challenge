class ComposeAugmentation:
    def __init__(self, transform=None, mode="train"):
        assert mode in ["train", "val", "test"]
        if mode != "train":
            assert transform
        self.mode = mode
        self.transform = transform

    def __call__(self, img, mask=None):
        if self.mode == "train":
            # Apply transform
            for f in self.transform:
                if mask is not None:
                    img, mask = f(img, mask)
                else:
                    img = f(img)
        if mask is not None:
            return img, mask
        else:
            return img
