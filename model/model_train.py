import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.hyper_parameters import hyper_paramaters as hp
from losses.GE2E_loss import GE2ELoss
from model.model_def import SpeechEmbedModel
from utils.data_load import SpeakerDatasetPreprocessed


def train_model(model_path):
    device = torch.device(hp.device)

    if hp.is_data_preprocessed:
        train_dataset = SpeakerDatasetPreprocessed()
    else:
        msg = "The project expects preprocessed spectrogram as an input."
        raise Exception(msg)

    train_loader = DataLoader(train_dataset,
                              batch_size=hp.training_N,  # number of speakers
                              shuffle=True,
                              num_workers=hp.training_num_workers,
                              drop_last=True)

    model = SpeechEmbedModel().to(device)
    if hp.restore_existing_model:
        model.load_state_dict(torch.load(model_path))

    ge2e_loss = GE2ELoss(device)
    # Both net and loss have trainable parameters
    lst_train_params_for = [
        {'params': model.parameters()},
        {'params': ge2e_loss.parameters()}
    ]
    optimizer = torch.optim.SGD(lst_train_params_for, lr=hp.lr)

    # creating the folder to save checkpoints
    os.makedirs(hp.checkpoint_dir, exist_ok=True)

    # setting model to train mode
    model.train()
    losses = []
    total_utterances = hp.training_N * hp.training_M
    for e in range(hp.training_epochs):
        epoch_st = time.time()
        batch_loss = []
        for mel_db_batch in train_loader:

            # sending data to GPU/TPU for calculation
            mel_db_batch = mel_db_batch.to(device)

            # mel is returned as 4x5x160x40 (batchxnum_speakerxutterlenxn_mel)and we will reshape it to 20x160x40
            new_shape = (total_utterances, mel_db_batch.size(2), mel_db_batch.size(3))
            mel_db_batch = torch.reshape(mel_db_batch, new_shape)

            perm = random.sample(range(0, total_utterances), total_utterances)
            unperm = list(perm)

            # saving the unpermutated status of the utterances, this will be used to fetch correct utterance per person
            for i, j in enumerate(perm):
                unperm[j] = i

            mel_db_batch = mel_db_batch[perm]

            # gradient accumulates
            optimizer.zero_grad()

            embeddings = model(mel_db_batch)
            embeddings = embeddings[unperm]

            # changing the shape back num_speakers x utter_per_speaker x embedding_vector
            embeddings = torch.reshape(embeddings, (hp.training_N, hp.training_M, embeddings.size(1)))

            # get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings)  # wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            batch_loss.append(loss.detach().numpy())

        curr_loss = np.mean(batch_loss)
        # list of all the losses for each epoch
        losses.append(curr_loss)

        # calc time elapsed
        epoch_et = time.time()
        hours, rem = divmod(epoch_et - epoch_st, 3600)
        minutes, seconds = divmod(rem, 60)
        time_msg = "{:0>2}:{:0>2}:{:0.0f}".format(int(hours), int(minutes), seconds)

        msg = "Epoch:[{0}/{1}] \tLoss:{2:.4f}\t Time:{3}"

        msg = msg.format(e + 1, hp.training_epochs, curr_loss, time_msg)
        print(msg)

        if (e + 1) % hp.log_interval == 0:
            if hp.log_file is not None:
                with open(hp.log_file, 'a') as f:
                    f.write(msg)

        if hp.checkpoint_dir is not None and (e + 1) % hp.checkpoint_interval == 0:
            model.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e + 1) + ".pth"
            ckpt_model_path = os.path.join(hp.checkpoint_dir, ckpt_model_filename)
            torch.save(model.state_dict(), ckpt_model_path)
            model.to(device).train()

    # save model
    model.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
    save_model_path = os.path.join(hp.checkpoint_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("Completed!! Trained model is saved at: \n", save_model_path)

    return model, losses


if __name__ == "__main__":
    if hp.is_training_mode:
        train_model(hp.model_path)
