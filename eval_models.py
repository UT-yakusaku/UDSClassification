import os
import sys
sys.path.append("./common")
import pickle
import torch
from common.infer import infer_models
from common.model import TransformerEncoderModel, RnnModel
from common.preprocessing import load_data, load_stft, load_spectrogram, select_data, preprocess

filenames = ['../RSC/20220729_2_1/20220729_2_1_data.abf', '../RSC/20220825_1_1/20220825_1_1_data.abf', '../RSC/20220925_2_1/20220925_2_1_data.abf', '../RSC/20220822_1_1/20220822_1_1_data.abf', '../RSC/20220922_2_1/20220922_2_1_data.abf', '../RSC/20220714_1_1/20220714_1_1_data.abf', '../RSC/20220723_2_3/20220723_2_3_data.abf', '../RSC/20220824_2_1/20220824_2_1_data.abf', '../RSC/20220925_1_1/20220925_1_1_data.abf', '../RSC/20220827_2_1/20220827_2_1_data.abf', '../RSC/20220921_1_1/20220921_1_1_data.abf', '../RSC/20220727_4_1/20220727_4_1_data.abf', '../RSC/20220928_4_1/20220928_4_1_data.abf', '../RSC/20220923_1_1/20220923_1_1_data.abf', '../RSC/20220912_1_1/20220912_1_1_data.abf', '../RSC/20220925_2_3/20220925_2_3_data.abf', '../RSC/20220824_3_1/20220824_3_1_data.abf', '../RSC/20220729_2_2/20220729_2_2_data.abf', '../RSC/20220926_2_2/20220926_2_2_data.abf', '../RSC/20220728_4_1/20220728_4_1_data.abf', '../RSC/20220724_1_1/20220724_1_1_data.abf', '../RSC/20220713_2_4/20220713_2_4_data.abf', '../RSC/20220723_2_1/20220723_2_1_data.abf', '../RSC/20220925_1_2/20220925_1_2_data.abf', '../RSC/20220714_1_4/20220714_1_4_data.abf', '../RSC/20220915_2_1/20220915_2_1_data.abf', '../RSC/20220928_4_2/20220928_4_2_data.abf']
stft_path = "../spectrogram/stft_fq128_hcf100.pkl"
# spectrogram_path = "../spectrogram/data_500_raw.hdf5"
label_fnames = []
for fname in filenames:
    if not os.path.exists(fname[:-4]+"_500_hmm_seg_50.npy"):
        continue
    label_fnames.append(fname[:-4]+"_500_hmm_seg_50.npy")
used_idx = [[0,940], [0, 1250], [0,700], [140,1020], [0, 510], [0,140], [0,360], [0,1230], [0,1210], 
            [0,0], [0,980], [0,900], [0,920], [0, 920], [0,0], [30,710], [50, 1180], [130,870],
            [0,400], [50,740], [0,0], [0,0], [0,830], [0,230], [0,790], [0,540], [0,0]]
id_to_no = [0,1,2,3,4,5,6,7,8,10,11,12,13,15,16,17,18,19,22,23,24,25]
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
pe = False
base_dir = f"../output/stft_transformers/6"
test_indices = list(range(15, 22))

all_data, label_data = load_data(filenames, label_fnames)
all_data = select_data(all_data, used_idx)
_, all_data = preprocess(all_data)
# all_data = load_stft(stft_path)
# all_data = load_spectrogram(spectrogram_path, id_to_no)
models = []
for i in range(3):
    ckpt_dir = base_dir + f"/{i+1}"
    ckpt_path = ckpt_dir + "/checkpoint.pth"
    model = TransformerEncoderModel(input_size=input_size, n_head=n_head, dim_ff=dim_ff, num_layers=num_layers, pe=pe, device=device)
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    models.append(model)
all_up_states, up_coins, down_coins = infer_models(models, label_data, all_data, test_indices, num_fq=num_fq, stride=stride, window_size=window_size, fs=fq_aft, device=device)
coins = {
    "up" : up_coins,
    "down" : down_coins
}
up_states = {
    "up_states" : all_up_states,
}
with open(base_dir + "/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(base_dir + "/up_states.pkl", "wb") as f:
    pickle.dump(up_states, f)

STDOUT = sys.stdout
sys.stdout = open(base_dir + "/test_coins.txt", "w")
for i,idx in enumerate(test_indices):
    print(f"no.{id_to_no[idx]+1}")
    print(f" up coin : {up_coins[i]}")
    print(f" down coin : {down_coins[i]}")
sys.stdout = STDOUT