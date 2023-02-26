import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
from tensorflow.keras import regularizers


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv1D(n_filters, 3, padding="same", activation="tanh", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv1D(n_filters, 3, padding="same", activation="tanh", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters, dropout_rate=.0):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool1D(2)(f)
    p = layers.Dropout(dropout_rate)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout_rate=.0):
    # upsample
    x = layers.Conv1DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(dropout_rate)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x



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
        output_sequence_length=312,
        vocabulary=vocabulary
    )(inputs1)

    x = keras.layers.Embedding(len(vocabulary), 350)(vectorize_layer)

    f1, p1 = downsample_block(x, 64, dropout_rate)
    f2, p2 = downsample_block(p1, 128, dropout_rate)
    f3, p3 = downsample_block(p2, 256, dropout_rate)

    bottleneck = double_conv_block(p3, 512)

    u1 = upsample_block(bottleneck, f3, 256, dropout_rate)

    u2 = upsample_block(u1, f2, 128, dropout_rate)

    u3 = upsample_block(u2, f1, 64, dropout_rate)

    time_input = keras.layers.Dense(32, activation=keras.activations.relu,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer)(layers.Concatenate()([inputs4, inputs5, inputs6, inputs7]))

    conc = layers.Concatenate()([keras.layers.Flatten()(u3), time_input])
    dense = keras.layers.Dense(64, activation=keras.activations.relu,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer)(conc)
    dense = keras.layers.Dropout(dropout_rate)(dense)

    dense = keras.layers.Dense(32, activation=keras.activations.relu,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer)(dense)
    dense = keras.layers.Dropout(dropout_rate)(dense)

    dense = keras.layers.Dense(16, activation=keras.activations.relu,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer)(layers.Concatenate()([dense, inputs2, inputs3]))
    dense = keras.layers.Dropout(dropout_rate)(dense)

    dense = keras.layers.Dense(16, activation=keras.activations.relu,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer)(dense)
    dense = keras.layers.Dropout(dropout_rate)(dense)

    output = keras.layers.Dense(6, activation=keras.activations.sigmoid)(dense)

    return keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7], outputs=output, name="unet12")