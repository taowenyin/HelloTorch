from torch.utils import data


class AbideData(data.Dataset):
    def __getitem__(self, index: int):
        return self.data_x[index], self.data_y[index]

    def __len__(self) -> int:
        return len(self.data_x)

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
