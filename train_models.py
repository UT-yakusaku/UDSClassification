import os
import sys
import pickle
sys.path.append("./common")
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.dataset import SpectrogramDataset
from common.infer import infer_model
from common.model import TransformerEncoderModel, RnnModel
from common.preprocessing import load_data, load_stft
from common.train import train_model
from common.visualize import visualize_loss


filenames = ['../RSC/20220729_2_1/20220729_2_1_data.abf', '../RSC/20220825_1_1/20220825_1_1_data.abf', '../RSC/20220925_2_1/20220925_2_1_data.abf', '../RSC/20220822_1_1/20220822_1_1_data.abf', '../RSC/20220922_2_1/20220922_2_1_data.abf', '../RSC/20220714_1_1/20220714_1_1_data.abf', '../RSC/20220723_2_3/20220723_2_3_data.abf', '../RSC/20220824_2_1/20220824_2_1_data.abf', '../RSC/20220925_1_1/20220925_1_1_data.abf', '../RSC/20220827_2_1/20220827_2_1_data.abf', '../RSC/20220921_1_1/20220921_1_1_data.abf', '../RSC/20220727_4_1/20220727_4_1_data.abf', '../RSC/20220928_4_1/20220928_4_1_data.abf', '../RSC/20220923_1_1/20220923_1_1_data.abf', '../RSC/20220912_1_1/20220912_1_1_data.abf', '../RSC/20220925_2_3/20220925_2_3_data.abf', '../RSC/20220824_3_1/20220824_3_1_data.abf', '../RSC/20220729_2_2/20220729_2_2_data.abf', '../RSC/20220926_2_2/20220926_2_2_data.abf', '../RSC/20220728_4_1/20220728_4_1_data.abf', '../RSC/20220724_1_1/20220724_1_1_data.abf', '../RSC/20220713_2_4/20220713_2_4_data.abf', '../RSC/20220723_2_1/20220723_2_1_data.abf', '../RSC/20220925_1_2/20220925_1_2_data.abf', '../RSC/20220714_1_4/20220714_1_4_data.abf', '../RSC/20220915_2_1/20220915_2_1_data.abf', '../RSC/20220928_4_2/20220928_4_2_data.abf']
stft_path = "../spectrogram/stft_fq64.pkl"
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
device = "cuda" if torch.cuda.is_available() else "cpu"
window_size = 50
stride = 25
num_layers = 3
input_size = 128
rnn_units = 64
n_head = 4
dim_ff = 256
epochs = 200
lr = 5e-5
batch_size = 128
pe = True
patience = 20
training_info = {
    "device" : device,
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
    "model" : "transformers",
    "data_path" : stft_path,
}
base_dir = f"../output/transformers/8"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

STDOUT = sys.stdout
sys.stdout = open(base_dir + "/log.txt", "w")

print(training_info)
_, label_data = load_data(filenames, label_fnames)
all_data = load_stft(stft_path)
all_up_coins = np.zeros(15)
all_down_coins = np.zeros(15)
for i in range(3):
    train_indices = [j for j in range(15) if (j < i*5) or (j >= (i+1)*5)]
    val_indices = [j for j in range(15) if (j >= i*5) and (j < (i+1)*5)]
    train_dataset = SpectrogramDataset(all_data, label_data, train_indices, window_size=window_size)
    val_dataset = SpectrogramDataset(all_data, label_data, val_indices, window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # model = RnnModel(num_layers=num_layers, input_size=input_size, rnn_units=rnn_units, rnn_type="gru")
    model = TransformerEncoderModel(input_size=input_size, n_head=n_head, dim_ff=dim_ff, num_layers=num_layers, pe=pe)
    print(model)

    ckpt_dir = base_dir + f"/{i+1}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = ckpt_dir + "/checkpoint.pth"
    train_losses, val_losses, _ = train_model(model, train_loader, val_loader, 
                                            epochs=epochs, lr=lr, patience=patience, path=ckpt_path, device=device)
    visualize_loss(train_losses, val_losses, f"loss : {i+1}", ckpt_dir+"/loss.png")
    with open(ckpt_dir+"/losses.pkl", "wb") as f:
        loss_dict = {
            "train_losses" : train_losses,
            "val_losses" : val_losses
        }
        pickle.dump(loss_dict, f)
    all_up_states, up_coins, down_coins = infer_model(model, label_data, all_data, val_indices, stride=stride, window_size=window_size, device=device)
    all_up_coins[i*5:(i+1)*5] = np.array(up_coins)
    all_down_coins[i*5:(i+1)*5] = np.array(down_coins)

for i in range(15):
    print(f"no.{id_to_no[i]+1}")
    print(f" {all_up_coins[i]}")
    print(f" {all_down_coins[i]}")

sys.stdout = STDOUT