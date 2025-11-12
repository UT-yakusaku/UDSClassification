import os
import sys
import pickle
sys.path.append("./common")
import torch
from torch.utils.data import DataLoader
from common.dataset import SpectrogramDataset
from common.infer import infer_model
from common.model import TransformerEncoderModel
from common.preprocessing import load_data, load_stft
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
# スペクトログラムファイルのパス
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
train_indices = list(range(10))
val_indices = list(range(10, 15))
device = "cuda" if torch.cuda.is_available() else "cpu"
window_size = 1000
stride = 500
num_layers = 3
input_size = 128
dim_ff = 256
epochs = 100
lr = 5e-5
patience = 20
training_info = {
    "device" : device,
    "window_size" : window_size,
    "stride" : stride,
    "num_layers" : num_layers,
    "input_size" : input_size,
    "dim_ff" : dim_ff,
    "epochs" : epochs,
    "lr" : lr,
    "patience" : patience,
    "model" : "transformers",
}
ckpt_dir = os.path.join(OUTPUT_DIR, CHECKPOINT_SUBDIR)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")

STDOUT = sys.stdout
sys.stdout = open(os.path.join(ckpt_dir, "log.txt"), "w")

_, label_data = load_data(filenames, label_fnames)
all_data = load_stft(stft_path)
train_dataset = SpectrogramDataset(all_data, label_data, train_indices, window_size=window_size)
val_dataset = SpectrogramDataset(all_data, label_data, val_indices, window_size=window_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = TransformerEncoderModel(num_layers=num_layers, input_size=input_size, dim_ff=dim_ff)
print(training_info)
print(model)

train_losses, val_losses, _ = train_model(
    model, train_loader, val_loader,
    epochs=epochs, lr=lr, patience=patience, path=ckpt_path, device=device
)
all_up_states, up_coins, down_coins = infer_model(
    model, label_data, all_data, val_indices,
    stride=stride, window_size=window_size, device=device
)

for i, idx in enumerate(val_indices):
    if idx < len(id_to_no):
        print(f"no.{id_to_no[idx]+1}")
    else:
        print(f"no.{idx+1}")
    print(f" {up_coins[i]}")
    print(f" {down_coins[i]}")

visualize_loss(train_losses, val_losses, "loss", os.path.join(ckpt_dir, "loss.png"))

with open(os.path.join(ckpt_dir, "losses.pkl"), "wb") as f:
    loss_dict = {
        "train_losses" : train_losses,
        "val_losses" : val_losses
    }
    pickle.dump(loss_dict, f)

sys.stdout = STDOUT