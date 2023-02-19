import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
import keras_nlp


class Model:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.name = "transformer_classifier1"
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
        dropout_rate = 0
        INTERMEDIATE_DIM = 100
        NUM_HEADS = 8
        # create vectorize layer, to transform words in integer
        with open('./Model/word_piece_vocabulary', 'r') as f:
            vocabulary = f.read().splitlines()
        x = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary, 500, lowercase=False,
                                                    strip_accents=False)(inputs1)
        # create embedding layer
        x = keras_nlp.layers.TokenAndPositionEmbedding(len(vocabulary), 500, 300)(x)

        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=dropout_rate
        )(inputs=x)
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=dropout_rate
        )(inputs=x)
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=dropout_rate
        )(inputs=x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)

        time_input = keras.layers.Dense(32, activation=keras.activations.relu)(
            layers.Concatenate()([inputs4, inputs5, inputs6, inputs7]))

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(
            layers.Concatenate()([x, keras.layers.Flatten()(time_input)]))

        dense = keras.layers.Dense(16, activation=keras.activations.relu)(
            layers.Concatenate()([dense, inputs2, inputs3]))

        dense = keras.layers.Dropout(dropout_rate)(dense)


        outputs = keras.layers.Dense(6, activation="sigmoid")(dense)

        return outputs

    def run_experiment(self, train_in, train_out, validation_in, validation_out, epochs=10, batch_size=100,
                       callbacks=None):

        if callbacks is None:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in, validation_out),
                                 epochs=epochs, batch_size=batch_size)
        else:
            res = self.model.fit(x=train_in, y=train_out, validation_data=(validation_in, validation_out),
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        return res

    def evaluate(self):
        pass
