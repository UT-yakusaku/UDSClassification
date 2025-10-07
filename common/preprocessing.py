import pyabf
import h5py
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import zscore


def load_data(data_files, label_files, verbose=False):
    all_data = []
    label_data = []
    if verbose:
        print("filename | shape")

    for dfile in data_files:
        abf = pyabf.ABF(dfile)
        data = abf.data[1]
        all_data.append(data)
        if verbose:
            print(dfile + " | " + data.shape)

    for lfile in label_files:
        label = np.load(lfile)
        label_data.append(label)

    return all_data, label_data


def select_data(all_data, used_idx, fq=20000):
    selected_data = []

    for i, data in enumerate(all_data):
        start, end = used_idx[i]
        if start == end:
            continue
        selected_data.append(data[fq*start:fq*end])

    return selected_data


def preprocess(all_data, fq_orig=20000, fq_aft=500, lcf=0.1, hcf=200):
    downsampled_data = []
    filtered_data = []
    niqf = fq_aft / 2
    times = fq_orig // fq_aft

    for data in all_data:
        data = np.mean(data[:len(data)//times*times].reshape(-1, times), axis=1)
        downsampled_data.append(data)
        b, a = butter(2, [lcf / niqf, hcf / niqf], btype="band")
        data = filtfilt(b, a, data)
        data = zscore(data)
        filtered_data.append(data)

    return downsampled_data, filtered_data


def load_spectrogram(path, id_to_no):
    f = h5py.File(path, "r")
    all_data = []
    for keyname in id_to_no:
        im_data = np.array(f[str(keyname)]["coef_imag"])
        re_data = np.array(f[str(keyname)]["coef_real"])
        data = np.vstack((im_data[None,:,:], re_data[None,:,:]))
        all_data.append(data)

    f.close()
    return all_data


def load_stft(path):
    with open(path, "rb") as f:
        all_data = pickle.load(f)

    return all_data