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
        self.name = "transformer_classifier2"
        inputs1 = keras.Input(shape=(1,), dtype=tf.float32)  # text
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
        INTERMEDIATE_DIM = 350
        NUM_HEADS = 2
        # create vectorize layer, to transform words in integer
        # vectorize_layer = keras.layers.TextVectorization(
        #     standardize='lower_and_strip_punctuation',
        #     split='whitespace',
        #     output_mode='int',
        #     output_sequence_length=256,
        #     vocabulary=self.vocabulary
        # )(inputs1)

        x = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=len(self.vocabulary),
            sequence_length=256,
            embedding_dim=300
        )(inputs1)

        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=dropout_rate
        )(inputs=x)

        x = keras.layers.Dense(256, activation=keras.activations.linear)(x)
        outputs = x

        return outputs

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
