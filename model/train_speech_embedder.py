import os
import random
import time

import torch
from torch.utils.data import DataLoader

from config.hyper_parameters import hyper_paramaters as hp
from data_related.data_load import Speaker_Dataset_Preprocessed
from model.speech_embedder_net import Speech_Embedder, GE2ELoss, get_centroids, get_cos_sim


def train_model(model_path):
    device = torch.device(hp.device)

    if hp.is_data_preprocessed:
        train_dataset = Speaker_Dataset_Preprocessed()
    else:
        msg = "The project expects preprocessed spectrogram as an input."
        raise Exception(msg)

    train_loader = DataLoader(train_dataset,
                              batch_size=hp.training_N,  # number of speakers
                              shuffle=True,
                              num_workers=hp.training_num_workers,
                              drop_last=True)

    model = Speech_Embedder().to(device)
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
    iteration = 0
    total_utterances = hp.training_N * hp.training_M
    for e in range(hp.training_epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)

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

            total_loss = total_loss + loss
            iteration += 1

            msg = "{0}\tEpoch:{1} [{2}/{3}], Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t"
            tot_batches = len(train_dataset) // hp.training_N
            t_loss = total_loss / (batch_id + 1)
            msg = msg.format(time.ctime(), e + 1, batch_id + 1, tot_batches, iteration, loss, t_loss)
            print(msg)
            if (batch_id + 1) % hp.log_interval == 0:
                if hp.log_file is not None:
                    with open(hp.log_file, 'a') as f:
                        f.write(msg)

        if hp.checkpoint_dir is not None and (e + 1) % hp.checkpoint_interval == 0:
            model.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".pth"
            ckpt_model_path = os.path.join(hp.checkpoint_dir, ckpt_model_filename)
            torch.save(model.state_dict(), ckpt_model_path)
            model.to(device).train()

    # save model
    model.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(hp.checkpoint_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def test(model_path):
    if hp.data_preprocessed:
        test_dataset = Speaker_Dataset_Preprocessed()
    else:
        msg = "The project expects preprocessed spectrogram as an input."
        raise Exception(msg)
    test_loader = DataLoader(test_dataset,
                             batch_size=hp.test_N,
                             shuffle=False,  # we dont need to shuffle for testing
                             num_workers=hp.test_num_workers,
                             drop_last=True)

    model = Speech_Embedder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert hp.test_M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (
                hp.test_N * hp.test_M // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (
                hp.test_N * hp.test_M // 2, verification_batch.size(2), verification_batch.size(3)))

            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = model(enrollment_batch)
            verification_embeddings = model(verification_batch)
            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test_N, hp.test_M // 2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (hp.test_N, hp.test_M // 2, verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cos_sim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01 * i + 0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(hp.test_N))])
                       / (hp.test_N - 1.0) / (float(hp.test_M / 2)) / hp.test_N)

                FRR = (sum([hp.test_M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(hp.test_N))])
                       / (float(hp.test_M / 2)) / hp.test_N)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


if __name__ == "__main__":
    if hp.is_training_mode:
        train_model(hp.model_path)
    else:
        test(hp.model_path)
