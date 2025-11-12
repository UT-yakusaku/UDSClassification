import numpy as np
import torch
from torch.utils.data import Dataset
from common.util import cal_stft


class SpectrogramDataset(Dataset):
    def __init__(self, all_data, label_data, data_indices, window_size=2000, transform=None):
        self.all_data = all_data
        self.label_data = label_data
        self.data_indices = data_indices
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.data_indices) * 200
    
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx//200]
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
    

class StftDataset(Dataset):
    def __init__(self, all_data, label_data, data_indices, num_fq=128, window_size=2000, transform=None):
        self.all_data = all_data
        self.label_data = label_data
        self.data_indices = data_indices
        self.window_size = window_size
        self.num_fq = num_fq
        self.sample_size = 1000
        self.transform = transform

    def __len__(self):
        return len(self.data_indices) * 2000
    
    def __getitem__(self, idx):
        data_idx = self.data_indices[idx//2000]
        data = self.all_data[data_idx]
        label = self.label_data[data_idx]

        time_len = data.shape[0]
        # max_start = time_len - self.window_size
        max_start = time_len - self.sample_size
        start = np.random.randint(0, max_start+1)
        # end = start + self.window_size
        end = start + self.sample_size
        spect_slice = np.array(data[start:end])
        spect_slice = cal_stft(spect_slice, num_fq=self.num_fq)
        spect_slice = spect_slice[:, :, -self.window_size:]
        # label_tensor = torch.tensor(label[start:end]).long()
        label_tensor = torch.tensor(label[end-self.window_size:end]).long()
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
    

class StftInferenceDataset(Dataset):
    def __init__(self, data, label, num_fq=128, window_size=2000, stride=500, transform=None):
        self.data = data
        self.label = label
        self.num_fq = num_fq
        self.window_size = window_size
        self.stride = stride
        self.sample_size = 1000
        self.transform = transform
        # self.lengths = (data.shape[0]-self.window_size) // self.stride + 1
        self.lengths = (self.data.shape[0]-self.sample_size) // self.stride + 1

    def __len__(self):
        return self.lengths
    
    def __getitem__(self, idx):
        start = idx * self.stride
        # end = start + self.window_size
        end = start + self.sample_size
        spect_slice = np.array(self.data[start:end])
        spect_slice = cal_stft(spect_slice, num_fq=self.num_fq)
        spect_slice = spect_slice[:, :, -self.window_size:]
        # label_tensor = torch.tensor(self.label[start:end]).long()
        label_tensor = torch.tensor(self.label[end-self.window_size:end]).long()
        spect_tensor = torch.from_numpy(spect_slice).float()

        if self.transform:
            spect_tensor = self.transform(spect_tensor)

        return spect_tensor, label_tensor