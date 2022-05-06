import numpy as np
import tensorflow as tf


def rnn_preprocess(x_train, x_test, y_train, y_test):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, x_test, y_train, y_test


def rnn_model(x_train):
    rnn1 = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    rnn = tf.keras.layers.Dropout(.35)(rnn1)
    rnn = tf.keras.layers.LSTM(3, return_sequences=True, recurrent_activation='sigmoid')(rnn)
    #rnn = tf.keras.layers.LSTM(256, recurrent_activation='sigmoid')(rnn)
    #rnn = tf.keras.layers.Dropout(.25)(rnn)
    dense1 = tf.keras.layers.Dense(512, activation='sigmoid')(rnn)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(512, activation='sigmoid')(dense1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    model = tf.keras.Model(inputs=[rnn1], outputs=outputs)
    return model


def rnn_train(model, x_train, x_test, y_train, y_test):
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=tf.keras.metrics.BinaryAccuracy())
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="models/logs/", histogram_freq=1)
    model.fit(x_train, y_train, epochs=1, callbacks=[tb_callback])

    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)
