import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample

# Example usage
data = np.random.rand(100, 10)
labels = np.random.randint(0, 2, size=(100,))

dataset = MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print(batch)