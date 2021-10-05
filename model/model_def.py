import torch
import torch.nn as nn

from config.hyper_parameters import hyper_paramaters as hp


class SpeechEmbedModel(nn.Module):

    def __init__(self):
        super(SpeechEmbedModel, self).__init__()

        # this creates a three stacks (hp.num_layer) of LSTM
        self.LSTM_stack = nn.LSTM(hp.n_mels, hp.hidden, num_layers=hp.num_layer, batch_first=True)

        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # feed forward layer
        self.projection = nn.Linear(hp.hidden, hp.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())

        # The embedding vector (d-vector) is defined as the L2 normalization of the network output
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
