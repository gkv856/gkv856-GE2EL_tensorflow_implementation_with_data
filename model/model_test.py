import random

import torch
from torch.utils.data import DataLoader

from config.hyper_parameters import hyper_paramaters as hp
from utils.data_load import Speaker_Dataset_Preprocessed
from losses.GE2E_loss import GE2ELoss
from model.model_def import SpeechEmbedModel


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

    model = SpeechEmbedModel()
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

            enrollment_centroids = GE2ELoss.get_centroids(enrollment_embeddings)

            sim_matrix = GE2ELoss.get_cos_sim(verification_embeddings, enrollment_centroids)

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
    if not hp.is_training_mode:
        test(hp.model_path)
