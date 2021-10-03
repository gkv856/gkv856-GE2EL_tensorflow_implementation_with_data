import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from tensorflow.keras.layers import LSTM, Input
from tensorflow import Variable

from gkv_implement.util import normalize, random_batch, calc_similarity, get_optimizer, calculate_loss
from string_constants.Constants import MODEL_SAVE_PATH
from string_constants.configuration_file import config


def create_model(inp_shape, m_name="model"):
    # input batch (time x batch x n_mel)
    inputs = Input(shape=inp_shape, dtype=tf.float32)
    x = inputs
    x = LSTM(units=config["embed_dim"], return_sequences=True, name="l1")(x)
    x = LSTM(units=config["embed_dim"], return_sequences=True, name="l2")(x)
    x = LSTM(units=config["embed_dim"], return_sequences=True, return_state=True, name="l3")(x)

    outputs = x
    model = tf.keras.Model(inputs, outputs, name=m_name)
    return model


def fit_train_model(model, epochs=10, m_save_path=MODEL_SAVE_PATH):
    # setting up the ground for training the model
    lr_factor = 1  # lr decay factor ( 1/2 per 10000 iteration)
    loss_acc = 0  # accumulated loss ( for running average of loss)

    # as per the paper, these variables helped in training better results
    w = Variable(10, name="w", dtype=tf.float32, trainable=True)
    b = Variable(-5, name="b", dtype=tf.float32, trainable=True)

    for epoch in range(epochs):

        now = datetime.now()
        fname = now.strftime("%d%m%y_%H%M%S")
        model_save_path = os.path.join(m_save_path, fname)
        lr = config["lr"] * lr_factor
        model_optimizer = get_optimizer(lr=lr)

        inputs = random_batch()
        # print(f"Input shape = {inputs.shape}")

        with tf.GradientTape(persistent=False) as tape:
            model_outputs = model(inputs)
            # print(f"model output shape = {model_outputs[0].shape}")

            # getting the first element of the output. the embedded d-vector
            out = model_outputs[0]
            out = normalize(out)
            # print(out.shape, type(out))

            # norm_embedded = normalize(embedded)
            N = out.shape[1]
            M = out.shape[2]
            P = out.shape[0]

            # calculating similarity matrix
            sim_matrix = calc_similarity(out, w, b, N, M, P)

            # calculating the current model loss
            curr_loss = calculate_loss(sim_matrix, N, M)

            # calculating the gradient
            m_gradients = tape.gradient(curr_loss, model.trainable_variables)

            # applying gradients to the trainable variable of the model (backpropagation)
            model_optimizer.apply_gradients(zip(m_gradients, model.trainable_variables))

        # handling task after each epoch
        # Training step, printing training steps
        if (epoch + 1) % 2 == 0:
            print('.', end='', flush=True)

        # accumulated loss for each 100 iteration and printing average loss per 100 epochs
        # then reset accumulated loss
        loss_acc += curr_loss
        if (epoch + 1) % 100 == 0:
            print("(Epoch : %d) loss: %.4f" % ((epoch + 1), loss_acc / 100))
            loss_acc = 0

        # saving model checkpoint after each 5k steps
        if (epoch + 1) % 5000 == 0:
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

        # reducing learning rate per 10k steps
        if (epoch + 1) % 10000 == 0:
            lr_factor /= 2
            print("learning rate is decayed! current lr : ", config["lr"] * lr_factor)

    return model


inp_shape = [config["speaker_num"] * config["utter_num"], config["n_mels"]]
model = create_model(inp_shape=inp_shape, m_name="model0")
trained_model = fit_train_model(model, 2)
