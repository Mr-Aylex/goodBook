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
        self.vocabulary = vocabulary
        self.name = "unet4"
        inputs1 = keras.Input(shape=(1,), dtype=tf.string)  # text
        inputs2 = keras.Input(shape=1, dtype=tf.float32)  # n_comment
        inputs3 = keras.Input(shape=1, dtype=tf.float32)  # n_votes

        layers = self.create_layers(inputs1, inputs2, inputs3)

        self.model = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=layers)

        self.model.compile(optimizer=keras.optimizers.Adamax(),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=[keras.metrics.categorical_accuracy,
                                    tfa.metrics.F1Score(num_classes=6, average='weighted')]
                           )

    def create_layers(self, inputs1, inputs2, inputs3):
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
        f4, p4 = downsample_block(p3, 512)

        bottleneck = double_conv_block(p4, 1024)

        u1 = upsample_block(bottleneck, f4, 512)
        u2 = upsample_block(u1, f3, 256)
        u3 = upsample_block(u2, f2, 128)
        u4 = upsample_block(u3, f1, 64)

        dense = keras.layers.Dense(64, activation=keras.activations.relu)(keras.layers.Flatten()(u4))
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(32, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        return keras.layers.Dense(6, activation=keras.activations.sigmoid)(dense)

    def run_experiment(self, data, output, epochs=10, batch_size=100, validation_split=0.2, callbacks=None):

        rating = keras.utils.to_categorical(output, num_classes=6)

        class_weight = self.get_class_weights(output)
        if callbacks is None:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split #, class_weight=class_weight
                                 )
        else:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, #class_weight=class_weight,
                                 callbacks=callbacks)
        return res

    def get_class_weights(self, output):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(output), y=output)
        di = {}
        for i in range(len(class_weights)):
            di[i] = class_weights[i]
        return di

    def evaluate(self):
        pass
