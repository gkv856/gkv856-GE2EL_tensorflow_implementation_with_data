import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from string_constants.configuration_file import config


def keyword_spot(spec, tdsv_frame):
    """ Keyword detection for data preprocess
        For VTCK data I truncate last 80 frames of trimmed audio - "Call Stella"
    :return: 80 frames spectrogram
    """
    return spec[:, -config["tdsv_frame"]:]


def random_batch(speaker_num=config["speaker_num"],
                 utter_num=config["utter_num"],
                 shuffle=True,
                 utter_start=0):
    """ Generate 1 batch.
        For TD-SV, noise is added to each utterance.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        noise_filenum : specify noise file or not (TD-SV)
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """

    # data path
    if config["train"]:
        path = config["train_path"]
    else:
        path = config["test_path"]


    # TI-SV
    np_file_list = os.listdir(path)
    total_speaker = len(np_file_list)

    if shuffle:
        # select random N speakers
        selected_files = random.sample(np_file_list, speaker_num)
    else:
        # select first N speakers
        selected_files = np_file_list[:speaker_num]

    utter_batch = []
    for file in selected_files:
        # load utterance spectrogram of selected speaker
        utters = np.load(os.path.join(path, file))
        if shuffle:
            # select M utterances per speaker
            # the speaker might have said 23 things.. meaning utter will be a list with 23 items in it
            # utter index will be a list of index, meaning, utter_batch will have 5 random statements from a speaker
            utter_index = np.random.randint(0, utters.shape[0], utter_num)

            # each speakers utterance [M, n_mels, frames] is appended
            # 5 statements of n_melsxframes size. meaning 5x40x180(since we set this length as tisv_frame=180 in config)
            utter_batch.append(utters[utter_index])
        else:
            utter_batch.append(utters[utter_start: utter_start + utter_num])

    # utterance batch [batch(NM), n_mels, frames]
    utter_batch = np.concatenate(utter_batch, axis=0)

    if config["train"]:
        # for train session, random slicing of input batch
        # instead of have a shape of 20x40x180, it will be 20x40xrandom_idx_bw_140_and_181
        frame_slice = np.random.randint(140, 181)
        utter_batch = utter_batch[:, :, :frame_slice]
    else:
        # for test session, fixed length slicing of input batch
        utter_batch = utter_batch[:, :, :160]

    # transpose [frames, batch, n_mels]
    # axis(2, 0, 1) means put axis =0 at 2nd index, axis=2 at 1st index and axis=1 means put 1st at 3rd
    # eg. (20, 40, 150) will become (150, 20, 40)
    utter_batch = np.transpose(utter_batch, axes=(2, 0, 1))

    return utter_batch


def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    normalization of following 3x4 list
      [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
      norm_factor for 1 element = [1^2 + 2^2 + 3^2 + 4^2] = [1 + 4 + 9 + 16]
      res for one item = [1/30, 2/30, 3/30, 4/30]
    """
    norm_factor = tf.reduce_sum(x ** 2, axis=-1, keepdims=True) + config["small_error"]
    norm_fac_sqrt = tf.sqrt(norm_factor)
    res = x / norm_fac_sqrt
    return res


def calc_cos_similarity(x, y, is_normalized=True):
    """ calculate similarity between tensors
    :return: cos similarity tf op node
    """
    if is_normalized:
        return tf.reduce_sum(x * y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x ** 2) + config["small_error"])
        y_norm = tf.sqrt(tf.reduce_sum(y ** 2) + config["small_error"])

        return tf.reduce_sum(x * y) / x_norm / y_norm


def calc_similarity(embedded, w, b, N=config["speaker_num"], M=config["utter_num"],
               P=config["embed_dim"], centroid_with_m_utter=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    reshaped_lstm_embedding = tf.reshape(embedded, shape=[N, M, P])
    # print(f"reshaped embedding {reshaped_lstm_embedding} \n and shape={reshaped_lstm_embedding.shape}")
    if centroid_with_m_utter is None:
        # [N, P] normalized centroid vectors eq.(1)
        # trying to find the centroid point for a speaker by taking the average
        # ck = Em[ekm] = 1/M for each m in M calc ekm. eq. (1)

        # For each input tuple, we compute the L2 normalized response
        # of the LSTM: {ej∼, (ek1, . . . , ekM )}. Here each e is an
        # embedding vector of ﬁxed dimension that results from the
        # sequence-to-vector mapping deﬁned by the LSTM.
        # The centroid of tuple (ek1, . . . , ekM ) represents the voiceprint
        # built from M utterances, and is deﬁned as follows:
        # ck = Em[ekm] = 1/M for each m in M calc ekm.

        # calculating centroid of all the utterances  as per eq. (1)
        eq1 = tf.reduce_mean(reshaped_lstm_embedding, axis=1)
        # print(f"eq1 = {eq1}")
        centroid_for_neg_similarity = normalize(eq1)
        # print(f"eq1 normalized = {centroid_for_neg_similarity}")
        # calculating centroid of true speaker eq. (8)
        tf_re_sum = tf.reduce_sum(reshaped_lstm_embedding, axis=1, keepdims=True)
        eq8 = tf.reshape(tf_re_sum - reshaped_lstm_embedding, shape=[N * M, P])
        centroid_for_true_similarity = normalize(eq8)

    # print("centroid_for_true_similarity", centroid_for_true_similarity)
    # print("centroid_for_neg_similarity", centroid_for_neg_similarity)
    # make similarity matrix eq.(9)

    final_lst = []
    for j in range(N):
        emb = reshaped_lstm_embedding[j, :, :]
        # print(f"j = {j}")
        # print(f"emb = {emb} ")
        new_lst = []
        for i in range(N):

            # if this is same speaker
            if i == j:
                matrix = centroid_for_true_similarity[i * M:(i + 1) * M, :] * emb

            # if comparing different speakers
            # here we are using eq. (9) to calculate negative similarities
            else:
                matrix = centroid_for_neg_similarity[i:(i + 1), :] * emb

            # reduced_sum is basically saying take a cosine difference
            t = tf.reduce_sum(matrix, axis=1, keepdims=True)
            new_lst.append(t)

        concated_lst = tf.concat(new_lst, axis=1)

        final_lst.append(concated_lst)

    S = tf.concat(final_lst, axis=0)

    # nested loop single line implementation of the same
    # S = tf.concat(
    # [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keepdims=True) if i==j
    #             else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keepdims=True) for i in range(N)],
    #             axis=1) for j in range(N)], axis=0)

    # eq. (10)
    S = tf.abs(w) * S + b

    return S


def get_optimizer(optimizer_name="sgd", lr=0.0001):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(lr)
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam(lr,
                                      beta1=config["beta1"],
                                      beta2=config["beta2"])
    else:
        raise AssertionError("Wrong optimizer type!")


def calculate_loss(S, type="softmax", N=config["speaker_num"], M=config["utter_num"]):
    """ calculate loss with similarity matrix(S) eq.(6) (7)
    :type: "softmax" or "contrast"
    :return: loss

      # We put a softmax on Sji,k for k = 1, . . . , N that
      # makes the output equal to 1 iff k = j, otherwise makes the out-
      # put equal to 0. Thus, the loss on each embedding vector eji could
      # be deﬁned as:
    """

    # colored entries in Fig.1
    S_correct = tf.concat([S[i * M:(i + 1) * M, i:(i + 1)] for i in range(N)], axis=0)

    # eq. (6)
    # L(eji) = −Sji,j + log ( for k 1 to N take exp(Sji,k)
    if type == "softmax":
        tf_log = tf.math.log(tf.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        tf_correct = S_correct - tf_log
        total = -tf.reduce_sum(tf_correct)


    # calculating eq. (7)
    # L(eji) = 1 − σ(Sji,j) + for max 1≤k≤N when k!=j σ(Sji)
    elif type == "contrast":

        S_sig = tf.sigmoid(S)

        concated_lst = []
        for i in range(N):
            new_lst = []
            for j in range(N):
                if i == j:
                    # if both are same person then return 0 matrix
                    eq7_p3 = 0 * S_sig[i * M:(i + 1) * M, j:(j + 1)]

                else:
                    eq7_p3 = S_sig[i * M:(i + 1) * M, j:(j + 1)]

                new_lst.append(eq7_p3)

            # concatenate the lst for each person
            eq7_p3_concat = tf.concat(new_lst, axis=1)
            concated_lst.append(eq7_p3_concat)

        eq7_p3_solved = tf.concat(concated_lst, axis=0)

        # total = tf.reduce_sum(1-tf.sigmoid(S_correct)+tf.reduce_max(S_sig, axis=1, keepdims=True))
        # this big formula is broken down as below
        eq7_p2 = tf.sigmoid(S_correct)
        eq7_p3_final = tf.reduce_max(eq7_p3_solved, axis=1, keepdims=True)
        val = 1 - eq7_p2 + eq7_p3_final
        total = tf.reduce_sum(val)

    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total
