import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv1D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv1D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool1D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv1DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


class Model:

    def __init__(self, vocabulary):
        self.train = None
        self.test = None
        self.vocabulary = vocabulary
        self.name = "unet6"
        inputs1 = keras.Input(shape=(1,), dtype=tf.string)  # text
        inputs2 = keras.Input(shape=1, dtype=tf.float32)  # n_comment
        inputs3 = keras.Input(shape=1, dtype=tf.float32)  # n_votes

        inputs4 = keras.Input(shape=1, dtype=tf.float32)  # read_at
        inputs5 = keras.Input(shape=1, dtype=tf.float32)  # date_added
        inputs6 = keras.Input(shape=1, dtype=tf.float32)  # date_updated
        inputs7 = keras.Input(shape=1, dtype=tf.float32)  # started_at

        layers = self.create_layers(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7)

        self.model = keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7], outputs=layers)

        self.model.compile(optimizer=keras.optimizers.Adamax(),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=[keras.metrics.categorical_accuracy,
                                    tfa.metrics.F1Score(num_classes=6, average='weighted')]
                           )

    def create_layers(self, inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7):
        regularizer = None
        dropout_rate = 0.15
        # create vectorize layer, to transform words in integer
        vectorize_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=512,
            vocabulary=self.vocabulary
        )(inputs1)

        x = layers.Embedding(len(self.vocabulary), 300)(vectorize_layer)

        f1, p1 = downsample_block(x, 64)
        f2, p2 = downsample_block(p1, 128)
        f3, p3 = downsample_block(p2, 256)

        bottleneck = double_conv_block(p3, 512)

        u1 = upsample_block(bottleneck, f3, 256)

        u2 = upsample_block(u1, f2, 128)

        u3 = upsample_block(u2, f1, 64)

        time_input = keras.layers.Dense(32, activation=keras.activations.relu)(layers.Concatenate()([inputs4, inputs5, inputs6, inputs7]))

        conc = layers.Concatenate()([layers.BatchNormalization()(keras.layers.Flatten()(u3)), time_input])
        dense = keras.layers.Dense(64, activation=keras.activations.relu)(conc)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(32, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(layers.Concatenate()([dense, inputs2, inputs3]))
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        return keras.layers.Dense(6, activation=keras.activations.sigmoid)(dense)

    def run_experiment(self, train_in, train_out, validation_in, validation_out, epochs=10, batch_size=100, callbacks=None):


        if callbacks is None:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in,  validation_out), epochs=epochs, batch_size=batch_size)
        else:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in,  validation_out), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        return res


    def evaluate(self):
        pass
