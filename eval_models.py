import os
import sys
sys.path.append("./common")
import pickle
import torch
from common.infer import infer_models
from common.model import TransformerEncoderModel, RnnModel
from common.preprocessing import load_data, load_stft, load_spectrogram, select_data, preprocess


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
# チェックポイントのベースディレクトリ（訓練時に使用した出力ディレクトリ）
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
window_size = 1000
stride = 500
num_layers = 3
input_size = 256
rnn_units = 64
n_head = 4
dim_ff = 512
pe = False
base_dir = os.path.join(OUTPUT_DIR, CHECKPOINT_SUBDIR)
test_indices = list(range(15, 22))

all_data, label_data = load_data(filenames, label_fnames)
all_data = select_data(all_data, used_idx)
_, all_data = preprocess(all_data)
# all_data = load_stft(stft_path)
# all_data = load_spectrogram(spectrogram_path, id_to_no)
models = []
for i in range(3):
    ckpt_dir = os.path.join(base_dir, str(i+1))
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    model = TransformerEncoderModel(input_size=input_size, n_head=n_head, dim_ff=dim_ff, num_layers=num_layers, pe=pe, device=device)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    models.append(model)
all_up_states, up_coins, down_coins = infer_models(
    models, label_data, all_data, test_indices,
    num_fq=num_fq, stride=stride, window_size=window_size,
    fs=fq_aft, device=device
)
coins = {
    "up" : up_coins,
    "down" : down_coins
}
up_states = {
    "up_states" : all_up_states,
}
with open(os.path.join(base_dir, "coins.pkl"), "wb") as f:
    pickle.dump(coins, f)
with open(os.path.join(base_dir, "up_states.pkl"), "wb") as f:
    pickle.dump(up_states, f)

STDOUT = sys.stdout
sys.stdout = open(os.path.join(base_dir, "test_coins.txt"), "w")
for i, idx in enumerate(test_indices):
    if idx < len(id_to_no):
        print(f"no.{id_to_no[idx]+1}")
    else:
        print(f"no.{idx+1}")
    print(f" up coin : {up_coins[i]}")
    print(f" down coin : {down_coins[i]}")
sys.stdout = STDOUT