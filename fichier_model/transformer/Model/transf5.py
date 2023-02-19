import numpy as np
import tensorflow as tf
from keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding
from keras_nlp.tokenizers import BytePairTokenizer

class Model:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.name = "transf5"
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
        kernel_regularizer = None  #regularizers.L1L2(l1=1e-7, l2=1e-6)
        bias_regularizer = None  # regularizers.L2(1e-6)
        activity_regularizer = None  # regularizers.L2(1e-7)
        dropout_rate = 0.15
        # create vectorize layer, to transform words in integer
        vectorize_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=300,
            vocabulary=self.vocabulary
        )(inputs1)
        embedding_output = TokenAndPositionEmbedding(len(self.vocabulary), 300, 300)(vectorize_layer)

        x = TransformerEncoder(500, 6, dropout_rate)(embedding_output)
        #x = TransformerEncoder(200, 6, dropout_rate)(x)

        encoder_out = keras.layers.Flatten()(x[:, 0:5, :])
        time_input = keras.layers.Dense(32, activation=keras.activations.relu,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer)(
            layers.Concatenate()([inputs4, inputs5, inputs6, inputs7]))

        conc = layers.Concatenate()([encoder_out, time_input])
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
                                   activity_regularizer=activity_regularizer)(
            layers.Concatenate()([dense, inputs2, inputs3]))
        dense = keras.layers.Dropout(dropout_rate)(dense)

        dense = keras.layers.Dense(16, activation=keras.activations.relu,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer)(dense)
        dense = keras.layers.Dropout(dropout_rate)(dense)
        return keras.layers.Dense(6, activation='softmax')(dense)

    def run_experiment(self, train_in, train_out, validation_in, validation_out, epochs=10, batch_size=100, callbacks=None):


        if callbacks is None:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in,  validation_out), epochs=epochs, batch_size=batch_size)
        else:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in,  validation_out), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        return res


    def evaluate(self):
        pass
