import os
import sys
import pickle
import h5py
sys.path.append("./common")
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.dataset import SpectrogramDataset, StftDataset
from common.infer import infer_model
from common.model import TransformerEncoderModel, RnnModel
from common.preprocessing import load_data, load_stft, load_spectrogram, select_data, preprocess
from common.train import train_model
from common.visualize import visualize_loss


# ========== パス設定 ==========
# データファイルのディレクトリ
DATA_DIR = "path/to/data"
# データファイル名のリスト（.abfファイルのパス）
# 例: ['path/to/data/file1_data.abf', 'path/to/data/file2_data.abf', ...]
filenames = [
    os.path.join(DATA_DIR, "file1_data.abf"),
    os.path.join(DATA_DIR, "file2_data.abf"),
    # 実際のファイルパスをここに追加してください
]
# スペクトログラムファイルのパス（使用しない場合はコメントアウト）
SPECTROGRAM_DIR = "path/to/spectrogram"
stft_path = os.path.join(SPECTROGRAM_DIR, "stft.pkl")
# ラベルファイルの拡張子パターン
LABEL_SUFFIX = "_label.npy"
# 出力ディレクトリ
OUTPUT_DIR = "path/to/output"
# チェックポイントのサブディレクトリ
CHECKPOINT_SUBDIR = "transformers"
# =============================

# ========== データ選択設定 ==========
# 各データファイルから使用する時間範囲（秒単位）を指定
# 形式: [[start1, end1], [start2, end2], ...]
# [start, end]が[0, 0]の場合はそのファイルをスキップ
# 例: [[0, 940], [0, 1250], [0, 700], ...]
used_idx = [
    [0, 0],  # file1_data.abfの使用範囲（秒単位）
    [0, 0],  # file2_data.abfの使用範囲（秒単位）
    # 実際の使用範囲をここに追加してください
]

# データインデックスをID番号にマッピングするリスト
# 例: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, ...]
# 出力時にid_to_no[idx]+1として使用される
id_to_no = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    # 実際のID番号マッピングをここに追加してください
]
# =============================

label_fnames = []
for fname in filenames:
    label_path = fname[:-4] + LABEL_SUFFIX
    if not os.path.exists(label_path):
        continue
    label_fnames.append(label_path)
fq_orig = 20000
fq_aft = 500
num_fq = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
window_size = 50
stride = 25
num_layers = 3
input_size = 256
rnn_units = 64
n_head = 4
dim_ff = 512
epochs = 200
lr = 5e-5
batch_size = 128
pe = False
patience = 50
optimizer_mode = "adam"
weight_decay = 0
training_info = {
    "device" : device,
    "num_fq" : num_fq,
    "window_size" : window_size,
    "stride" : stride,
    "num_layers" : num_layers,
    "input_size" : input_size,
    "dim_ff" : dim_ff,
    "n_head" : n_head,
    "pe" : pe,
    # "rnn_units" : rnn_units,
    "epochs" : epochs,
    "lr" : lr,
    "batch_size" : batch_size,
    "patience" : patience,
    "weight_decay" : weight_decay,
    "optimizer" : optimizer_mode,
    "model" : "transformers",
    "data_path" : stft_path,
    # "data_path" : spectrogram_path,
}
base_dir = os.path.join(OUTPUT_DIR, CHECKPOINT_SUBDIR)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

STDOUT = sys.stdout
sys.stdout = open(os.path.join(base_dir, "log.txt"), "w")

print(training_info)
all_data, label_data = load_data(filenames, label_fnames)
all_data = select_data(all_data, used_idx, fq=fq_orig)
_, all_data = preprocess(all_data, fq_orig=fq_orig, fq_aft=fq_aft)
# all_data = load_stft(stft_path)
# all_data = load_spectrogram(spectrogram_path, id_to_no)
all_up_coins = np.zeros(15)
all_down_coins = np.zeros(15)
for i in range(3):
    train_indices = [j for j in range(15) if (j < i*5) or (j >= (i+1)*5)]
    val_indices = [j for j in range(15) if (j >= i*5) and (j < (i+1)*5)]
    # train_dataset = SpectrogramDataset(all_data, label_data, train_indices, window_size=window_size)
    # val_dataset = SpectrogramDataset(all_data, label_data, val_indices, window_size=window_size)
    train_dataset = StftDataset(all_data, label_data, train_indices, num_fq=num_fq, window_size=window_size)
    val_dataset = StftDataset(all_data, label_data, val_indices, num_fq=num_fq, window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # model = RnnModel(num_layers=num_layers, input_size=input_size, rnn_units=rnn_units, rnn_type="gru")
    model = TransformerEncoderModel(input_size=input_size, n_head=n_head, dim_ff=dim_ff, num_layers=num_layers, pe=pe, device=device)
    print(model)
    model.to(device)

    ckpt_dir = os.path.join(base_dir, str(i+1))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    train_losses, val_losses, _ = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, patience=patience,
        weight_decay=weight_decay, optimizer_mode=optimizer_mode,
        path=ckpt_path, device=device
    )
    visualize_loss(train_losses, val_losses, f"loss : {i+1}", os.path.join(ckpt_dir, "loss.png"))
    with open(os.path.join(ckpt_dir, "losses.pkl"), "wb") as f:
        loss_dict = {
            "train_losses" : train_losses,
            "val_losses" : val_losses
        }
        pickle.dump(loss_dict, f)
    all_up_states, up_coins, down_coins = infer_model(
        model, label_data, all_data, val_indices,
        num_fq=num_fq, stride=stride, window_size=window_size, device=device
    )
    all_up_coins[i*5:(i+1)*5] = np.array(up_coins)
    all_down_coins[i*5:(i+1)*5] = np.array(down_coins)

for i in range(15):
    if i < len(id_to_no):
        print(f"no.{id_to_no[i]+1}")
    else:
        print(f"no.{i+1}")
    print(f" {all_up_coins[i]}")
    print(f" {all_down_coins[i]}")

sys.stdout = STDOUT