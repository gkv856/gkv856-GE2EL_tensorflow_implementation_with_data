import torch
import torch.nn as nn

from config.hyper_parameters import hyper_paramaters as hp
from utils.utils import get_centroids, calc_loss, get_cos_sim


class Speech_Embedder(nn.Module):

    def __init__(self):
        super(Speech_Embedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.n_mels, hp.hidden, num_layers=hp.num_layer, batch_first=True)

        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.projection = nn.Linear(hp.hidden, hp.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        # Clamps all elements in input into the range [ min, max ]
        # https://pytorch.org/docs/stable/generated/torch.clamp.html
        torch.clamp(self.w, hp.small_err)
        centroids = get_centroids(embeddings)
        cos_similarity = get_cos_sim(embeddings, centroids)
        sim_matrix = self.w * cos_similarity.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss
