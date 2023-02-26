import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
from tensorflow.keras import regularizers


def model(vocabulary, dropout_rate=0.15, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
          bias_regularizer=regularizers.L2(1e-4), activity_regularizer=regularizers.L2(1e-5)):
    inputs1 = keras.Input(shape=(1,), dtype=tf.string)  # text
    inputs2 = keras.Input(shape=1, dtype=tf.float32)  # n_comment
    inputs3 = keras.Input(shape=1, dtype=tf.float32)  # n_votes

    inputs4 = keras.Input(shape=1, dtype=tf.float32)  # read_at
    inputs5 = keras.Input(shape=1, dtype=tf.float32)  # date_added
    inputs6 = keras.Input(shape=1, dtype=tf.float32)  # date_updated
    inputs7 = keras.Input(shape=1, dtype=tf.float32)  # started_at

    # create vectorize layer, to transform words in integer
    vectorize_layer = keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        output_mode='int',
        output_sequence_length=352,
        vocabulary=vocabulary
    )(inputs1)


    x = keras.layers.Embedding(len(vocabulary), output_dim=64, mask_zero=False)(vectorize_layer)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    output = keras.layers.Dense(6, activation=tf.keras.activations.softmax)(x)
    return keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7], outputs=output,
                       name="lstm5")

