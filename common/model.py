import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnRnnUDSModel(nn.Module):
    def __init__(self, freq_bins=256, time_steps=2000, conv_filters=32, rnn_units=64, rnn_type="lstm"):
        super(CnnRnnUDSModel, self).__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=conv_filters,
                kernel_size=(5, 5),
                padding="same"
            ),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, freq_bins, time_steps)
            cnn_output_shape = self.cnn_block(dummy_input).shape
            rnn_input_size = cnn_output_shape[1] * cnn_output_shape[2]

        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_units,
                num_layers=1,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_units,
                num_layers=1,
                batch_first=True
            )

        self.output_layer = nn.Linear(rnn_units, 1)

    def forward(self, x):
        x_cnn = self.cnn_block(x)
        x_permuted = x_cnn.permute(0,3,1,2)
        batch_size = x_permuted.size(0)
        time_steps = x_permuted.size(1)
        x_reshaped = x_permuted.reshape(batch_size, time_steps, -1)

        rnn_output, _ = self.rnn(x_reshaped)
        output = self.output_layer(rnn_output)

        return output.squeeze()