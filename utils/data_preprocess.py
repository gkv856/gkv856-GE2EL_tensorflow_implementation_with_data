import os

import librosa
import numpy as np

from config.hyper_parameters import hyper_paramaters as hp


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is split by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """
    print("Text independent speaker verification (TISV) utterance feature extraction started..")

    # Create folder if does not exist, if exist then ignore
    os.makedirs(hp.train_spectrogram_path, exist_ok=True)
    os.makedirs(hp.test_spectrogram_path, exist_ok=True)

    # lower bound of utterance length
    utter_min_len = (hp.tisv_frame * hp.hop + hp.window) * hp.sr

    lst_folders = os.listdir(hp.raw_data)
    total_speaker_num = len(lst_folders)
    train_speaker_num = (total_speaker_num // 10) * 8  # split total data 90% train and 10% test

    print("Total speakers : %d" % total_speaker_num)
    print("Train : %d, Test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))

    for i, folder in enumerate(os.listdir(hp.raw_data)):
        # path of each speaker
        speaker_path = os.path.join(hp.raw_data, folder)
        print(f"Processing speaker no. {i + 1}")
        utterances_spec = []
        k = 0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)  # path of each utterance
            utter, sr = librosa.core.load(utter_path, hp.sr)  # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)  # voice activity detection
            for interval in intervals:
                if (interval[1] - interval[0]) >= utter_min_len:  # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]: interval[1]]  # save first and last 180 frames of spectrogram.

                    dict_data = {
                        "y": utter_part,
                        "n_fft": hp.nfft,
                        "win_length": int(hp.window * sr),
                        "hop_length": int(hp.hop * sr)
                    }
                    # this return S as complex number with magnitude and direction
                    S = librosa.core.stft(**dict_data)

                    # we take ABS of complex number, we get the magnitude and we loose the direction info
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.nfft, n_mels=hp.n_mels)
                    S = np.log10(np.dot(mel_basis, S) + hp.small_err)  # log mel spectrogram of utterances

                    # first 180 frames of partial utterance
                    audio_extract = S[:, :hp.tisv_frame]
                    utterances_spec.append(audio_extract)

                    # last 180 frames of partial utterance
                    if (interval[1] - interval[0]) > utter_min_len:
                        audio_extract = S[:, -hp.tisv_frame:]
                        utterances_spec.append(audio_extract)

        utterances_spec = np.array(utterances_spec)

        # Checking if speaker's utterance qualifies to be used. i.e. a min utterance length is available in the audio
        if utterances_spec.shape[0] > 0:
            print(utterances_spec.shape)
            if i < train_speaker_num:  # save spectrogram as numpy file
                file_full_path = os.path.join(hp.train_spectrogram_path, f"speaker{i + 1}.npy")
            else:
                file_full_path = os.path.join(hp.test_spectrogram_path, f"speaker{i + 1}.npy")

            np.save(file_full_path, utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()