import math
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import stft
import pywt


class EarlyStopping:
    def __init__(self, patience=50, path="checkpoint.pth"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def cal_stft(data, fq=500, num_fq=64):
    nperseg = min(len(data), int(num_fq*5))
    _, _, Zxx = stft(data, fs=fq, nperseg=nperseg, noverlap=nperseg-1)
    Zxx = Zxx[:num_fq, :len(data)]
    re_data = np.real(Zxx).astype(np.float32)
    im_data = np.imag(Zxx).astype(np.float32)
    result = np.vstack((im_data[None,:,:], re_data[None,:,:]))
    return result


def cal_spectrogram(data, fs=500, num_fq=256):
    lcf, hcf = 0.1, 200
    fq = np.exp(np.linspace(np.log(lcf), np.log(hcf), num_fq))

    Fc = pywt.central_frequency("cmor1.5-2")
    scl = fs * Fc / fq
    coef, _ = pywt.cwt(data, scl, "cmor1.5-2")
    re_data = np.real(coef).astype(np.float32)
    im_data = np.imag(coef).astype(np.float32)
    result = np.vstack((im_data[None,:,:], re_data[None,:,:]))
    return result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.to(device)
        self.pe = pe

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
        