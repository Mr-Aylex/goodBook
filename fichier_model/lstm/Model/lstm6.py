import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa

class Model:

    def __init__(self, vocabulary):
        self.train = None
        self.test = None
        self.vocabulary = vocabulary
        self.name = "lstm6"
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
            output_sequence_length=256,
            vocabulary=self.vocabulary
        )(inputs1)

        x = layers.Embedding(len(self.vocabulary), 300)(vectorize_layer)

        x = layers.Bidirectional(layers.LSTM(300, return_sequences=True))(x)
        x = layers.LSTM(300, return_sequences=False)(x)

        time_input = keras.layers.Dense(32, activation=keras.activations.relu)(layers.Concatenate()([inputs4, inputs5, inputs6, inputs7]))

        conc = layers.Concatenate()([keras.layers.Flatten()(x), time_input])
        dense = keras.layers.Dense(64, activation=keras.activations.relu)(conc)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(32, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(layers.Concatenate()([dense, inputs2, inputs3]))
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)

        return keras.layers.Dense(6, activation=keras.activations.sigmoid)(dense)

    def run_experiment(self, data, output, epochs=10, batch_size=100, validation_split=0.2, callbacks=None):

        rating = keras.utils.to_categorical(output, num_classes=6)


        if callbacks is None:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        else:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=validation_split)
        return res


    def evaluate(self):
        pass
