import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SpectrogramInferenceDataset, StftInferenceDataset


def infer_model(model, label_data, all_data, test_indices, num_fq=128, stride=500, window_size=2000, fs=500, device="cuda"):
    model.eval()
    all_up_states = []
    up_coins = np.zeros(len(test_indices))
    down_coins = np.zeros(len(test_indices))
    sample_size = 1000
    for i, test_idx in enumerate(test_indices):
        label = label_data[test_idx]
        data = all_data[test_idx]
        # up_states_count = np.zeros(data.shape[2])
        up_states_count = np.zeros(data.shape[0]-sample_size+window_size)
        # overlap_count = np.zeros(data.shape[2])
        overlap_count = np.zeros(data.shape[0]-sample_size+window_size)
        # test_dataset = SpectrogramInferenceDataset(data, label, window_size=window_size, stride=stride)
        test_dataset = StftInferenceDataset(data, label, num_fq=num_fq, window_size=window_size, stride=stride)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        cnt = 0
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).cpu().detach().numpy()
            pstates = outputs >= 0
            for pstate in pstates:
                start = cnt*stride
                end = start + window_size
                up_states_count[start:end] += pstate.astype(int)
                overlap_count[start:end] += 1
                cnt += 1

        up_states = (up_states_count / overlap_count) >= 0.5
        up_states = up_states.astype(int)

        # filtering
        t = 0.04
        min_duration_samples = t * fs
        up_states_downfiltered = np.ones(len(up_states), dtype=int)
        up_transitions = np.where(np.diff(up_states.astype(int)) == 1)[0]
        down_transitions = np.where(np.diff(up_states.astype(int)) == -1)[0]
        if down_transitions[0] > up_transitions[0]:
            up_states_downfiltered[:up_transitions[0]] = 0
            up_transitions = up_transitions[1:]
        for start, end in zip(down_transitions, up_transitions):
            if end - start >= min_duration_samples:
                up_states_downfiltered[start:end] = 0
        up_states = up_states_downfiltered
        up_states_upfiltered = np.zeros_like(up_states, dtype=int)
        up_transitions = np.where(np.diff(up_states.astype(int)) == 1)[0]
        down_transitions = np.where(np.diff(up_states.astype(int)) == -1)[0]
        if up_transitions[0] > down_transitions[0]:
            up_states_upfiltered[:down_transitions[0]] = 1
            down_transitions = down_transitions[1:]
        for start, end in zip(up_transitions, down_transitions):
            if end - start >= min_duration_samples:
                up_states_upfiltered[start:end] = 1
        up_states = up_states_upfiltered
        
        all_up_states.append(up_states)

        label = label[sample_size-window_size:]
        up_coin = sum(np.where(label + up_states >= 2)[0]) / sum(np.where(label + up_states >= 1)[0])
        down_coin = sum(np.where(label + up_states <= 0)[0]) / sum(np.where(label + up_states <= 1)[0])
        up_coins[i] = up_coin
        down_coins[i] = down_coin

    return all_up_states, up_coins, down_coins


def infer_models(models, label_data, all_data, test_indices, num_fq=128, stride=500, window_size=2000, fs=500, device="cuda"):
    all_up_states = []
    up_coins = np.zeros(len(test_indices))
    down_coins = np.zeros(len(test_indices))
    sample_size = 1000
    for i, test_idx in enumerate(test_indices):
        label = label_data[test_idx]
        data = all_data[test_idx]
        # test_dataset = SpectrogramInferenceDataset(data, label, window_size=window_size, stride=stride)
        test_dataset = StftInferenceDataset(data, label, num_fq=num_fq, window_size=window_size, stride=stride)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # all_ustates = np.zeros(data.shape[2])
        all_ustates = np.zeros(data.shape[0]-sample_size+window_size)
        for model in models:
            model.eval()
            cnt = 0
            # up_states_count = np.zeros(data.shape[2])
            up_states_count = np.zeros(data.shape[0]-sample_size+window_size)
            # overlap_count = np.zeros(data.shape[2])
            overlap_count = np.zeros(data.shape[0]-sample_size+window_size)
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).cpu().detach().numpy()
                pstates = outputs >= 0
                for pstate in pstates:
                    start = cnt*stride
                    end = start + window_size
                    up_states_count[start:end] += pstate.astype(int)
                    overlap_count[start:end] += 1
                    cnt += 1

            up_states = (up_states_count / overlap_count) >= 0.5
            up_states = up_states.astype(int)

            # filtering
            t = 0.04
            min_duration_samples = t * fs
            up_states_downfiltered = np.ones(len(up_states), dtype=int)
            up_transitions = np.where(np.diff(up_states.astype(int)) == 1)[0]
            down_transitions = np.where(np.diff(up_states.astype(int)) == -1)[0]
            if down_transitions[0] > up_transitions[0]:
                up_states_downfiltered[:up_transitions[0]] = 0
                up_transitions = up_transitions[1:]
            for start, end in zip(down_transitions, up_transitions):
                if end - start >= min_duration_samples:
                    up_states_downfiltered[start:end] = 0
            up_states = up_states_downfiltered
            up_states_upfiltered = np.zeros_like(up_states, dtype=int)
            up_transitions = np.where(np.diff(up_states.astype(int)) == 1)[0]
            down_transitions = np.where(np.diff(up_states.astype(int)) == -1)[0]
            if up_transitions[0] > down_transitions[0]:
                up_states_upfiltered[:down_transitions[0]] = 1
                down_transitions = down_transitions[1:]
            for start, end in zip(up_transitions, down_transitions):
                if end - start >= min_duration_samples:
                    up_states_upfiltered[start:end] = 1
            up_states = up_states_upfiltered
            all_ustates += up_states
            
        up_states = all_ustates > (len(models) / 2)
        up_states = up_states.astype(int)
        all_up_states.append(up_states)

        label = label[sample_size-window_size:]
        up_coin = sum(np.where(label + up_states >= 2)[0]) / sum(np.where(label + up_states >= 1)[0])
        down_coin = sum(np.where(label + up_states <= 0)[0]) / sum(np.where(label + up_states <= 1)[0])
        up_coins[i] = up_coin
        down_coins[i] = down_coin

    return all_up_states, up_coins, down_coins