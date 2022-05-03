from src.data_preprocessing import *
from src.models import *
import tensorflow as tf

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

#====================
#data load
#====================

df = load_data(training_path, label_path)
df = dummy_creation(df)
df = normalize_float(df)
X_train, X_test, y_train, y_test = split_data(df, only_track= True)

def rnn_preprocess(X_train, X_test, y_train, y_test):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, y_train, y_test


def rnn_model(X_train):
    rnn1 = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    rnn = tf.keras.layers.Dropout(.35)(rnn1)
    rnn = tf.keras.layers.LSTM(256, return_sequences=True, recurrent_activation='sigmoid')(rnn)
    rnn = tf.keras.layers.LSTM(256, recurrent_activation='sigmoid')(rnn)
    rnn = tf.keras.layers.Dropout(.25)(rnn)
    dense1 = tf.keras.layers.Dense(512, activation='sigmoid')(rnn)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(512, activation='sigmoid')(dense1)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=[rnn1], outputs=outputs)
    return model


def rnn_train(model, X_train, X_test, y_train, y_test):
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    model.fit(X_train, y_train, epochs=10)

    results = model.evaluate(X_test, y_test, batch_size=128)
    print(results)


X_train, X_test, y_train, y_test = split_data(df, only_track=False)

X_train, X_test, y_train, y_test = rnn_preprocess(X_train, X_test, y_train, y_test)
model = rnn_model(X_train)
rnn_train(model, X_train, X_test, y_train, y_test)