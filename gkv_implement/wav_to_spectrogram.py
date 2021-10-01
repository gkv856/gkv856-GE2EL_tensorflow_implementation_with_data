import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from string_constants.Constants import AUDIO_FOLDER, TRAIN_SPEC_FOLDER, TEST_SPEC_FOLDER
from string_constants.configuration_file import config


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is split by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")

    # lower bound of utterance length
    utter_min_len = (config["tisv_frame"] * config["hop"] + config["window"]) * config["sr"]
    total_speaker_num = len(os.listdir(AUDIO_FOLDER))
    train_speaker_num = (total_speaker_num // 10) * 9  # split total data 90% train and 10% test
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))
    for i, folder in enumerate(os.listdir(AUDIO_FOLDER)):
        speaker_path = os.path.join(AUDIO_FOLDER, folder)  # path of each speaker
        print(f"{i+1} th speaker processing...")
        utterances_spec = []
        k = 0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)  # path of each utterance
            utter, sr = librosa.core.load(utter_path, config["sr"])  # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)  # voice activity detection
            for interval in intervals:
                if (interval[1] - interval[0]) >= utter_min_len:  # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]: interval[1]]  # save first and last 180 frames of spectrogram.

                    dict_data = {
                        "y": utter_part,
                        "n_fft": config["nfft"],
                        "win_length": int(config["window"] * sr),
                        "hop_length": int(config["hop"] * sr)
                    }
                    # this return S as complex number with magnitude and direction
                    S = librosa.core.stft(**dict_data)

                    # we take ABS of complex number, we get the magnitude and we loose the direction info
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config["sr"], n_fft=config["nfft"], n_mels=config["n_mels"])
                    S = np.log10(np.dot(mel_basis, S) + config["small_error"])  # log mel spectrogram of utterances

                    # first 180 frames of partial utterance
                    audio_extract = S[:, :config["tisv_frame"]]
                    utterances_spec.append(audio_extract)

                    # last 180 frames of partial utterance
                    if (interval[1] - interval[0]) > utter_min_len:
                        audio_extract = S[:, -config["tisv_frame"]:]
                        utterances_spec.append(audio_extract)

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(TRAIN_SPEC_FOLDER, f"speaker{i+1}.npy"), utterances_spec)
        else:
            np.save(os.path.join(TEST_SPEC_FOLDER, f"speaker{i+1-train_speaker_num}.npy"), utterances_spec)


save_spectrogram_tisv()
