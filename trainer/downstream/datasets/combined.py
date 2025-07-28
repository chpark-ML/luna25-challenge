from torch.utils.data import ConcatDataset


class CombinedDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
