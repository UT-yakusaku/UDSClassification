import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, all_data, label_data, data_indices, window_size=2000, transform=None):
        self.all_data = all_data
        self.label_data = label_data
        self.data_indices = data_indices
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.data_indices) * 100
    
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx//100]
        data = self.all_data[data_idx]
        label = self.label_data[data_idx]

        time_len = data.shape[2]
        max_start = time_len - self.window_size
        start = np.random.randint(0, max_start+1)
        end = start + self.window_size
        spect_slice = np.array(data[:, :, start:end])
        label_tensor = torch.tensor(label[start:end]).long()
        spect_tensor = torch.from_numpy(spect_slice).float()

        if self.transform:
            spect_tensor = self.transform(spect_tensor)

        return spect_tensor, label_tensor
    

class SpectrogramInferenceDataset(Dataset):
    def __init__(self, data, label, window_size=2000, stride=500, transform=None):
        self.data = data
        self.label = label
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.lengths = (data.shape[2]-self.window_size) // self.stride + 1

    def __len__(self):
        return self.lengths
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        spect_slice = np.array(self.data[:, :, start:end])
        label_tensor = torch.tensor(self.label[start:end]).long()
        spect_tensor = torch.from_numpy(spect_slice).float()

        if self.transform:
            spect_tensor = self.transform(spect_tensor)

        return spect_tensor, label_tensor