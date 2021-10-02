import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.layers import LSTM, Input
from tensorflow import Variable

from gkv_implement.util import normalize, random_batch, calc_similarity, get_optimizer, calculate_loss
from string_constants.configuration_file import config

model_optimizer = get_optimizer(lr=config["lr"])


def create_model(inp_shape, m_name="model"):
    # input batch (time x batch x n_mel)
    inputs = Input(shape=inp_shape, dtype=tf.float32)
    x = inputs
    x = LSTM(units=config["embed_dim"], return_sequences=True, name="l1")(x)
    x = LSTM(units=config["embed_dim"], return_sequences=True, name="l2")(x)
    x = LSTM(units=config["embed_dim"], return_sequences=True, name="l3")(x)
    embedded = x[-1]
    norm_embedding = normalize(embedded)
    print("norm_embedding embedded size: ", norm_embedding.shape)

    model = tf.keras.Model(inputs, outputs=norm_embedding, name=m_name)
    return model


@tf.function
def fit_train_model(model, epochs=10):
    w = Variable(10, name="w")
    b = Variable(-5, name="b")

    for epoch in range(epochs):
        inputs = random_batch()
        print(f"Input shape = {inputs.shape}")

        with tf.GradientTape(persistent=True) as tape:
            model_outputs = model(inputs)
            print(f"model output shape = {model_outputs[0].shape}")
            out = model_outputs[0]
            print(out.shape, type(out))
            # embedded = model_outputs[-1]  # the last ouput is the embedded d-vector
            # norm_embedded = normalize(embedded)
            N = out.shape[1]
            M = out.shape[2]
            P = out.shape[0]
            # reshaped_lstm_embedding = tf.reshape(out, shape=[out.shape[1], out.shape[2], out.shape[0]])

            reshaped_lstm_embedding = tf.reshape(out, shape=[N, M, P])
            print(reshaped_lstm_embedding.shape)

            sim_matrix = calc_similarity(out, w, b, N, M, P)

            print("similarity matrix size: ", sim_matrix.shape)
            loss = calculate_loss(sim_matrix, type=config["loss"])

            m_gradients = tape.gradient(loss, model.trainable_variables)

            model_optimizer.apply_gradients(zip(m_gradients, model.trainable_variables))

        # Training step
        if (epoch + 1) % 2 == 0:
            print('.', end='', flush=True)
