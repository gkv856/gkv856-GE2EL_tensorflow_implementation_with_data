import os
import random

import numpy as np
import torch
from config.hyper_parameters import hyper_paramaters as hp
from torch.utils.data import Dataset


class Speaker_Dataset_Preprocessed(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.is_training_mode:
            self.path = hp.train_spectrogram_path
            self.utter_num = hp.training_M
        else:
            self.path = hp.test_spectrogram_path
            self.utter_num = hp.test_M

        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            # select random speaker
            selected_file = random.sample(np_file_list, 1)[0]
        else:
            selected_file = np_file_list[idx]

        # load utterance spectrogram of selected speaker
        utters = np.load(os.path.join(self.path, selected_file))
        if self.shuffle:
            # select M utterances per speaker
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)
            utterance = utters[utter_index]
        else:
            # utterances of a speaker [batch(M), n_mels, frames]
            utterance = utters[self.utter_start: self.utter_start + self.utter_num]

        # TODO implement variable length batch size
        utterance = utterance[:, :, :160]

        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
        return utterance
