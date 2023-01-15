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

        layers = self.create_layers(inputs1, inputs2, inputs3)

        self.model = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=layers)

        self.model.compile(optimizer=keras.optimizers.Adamax(),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=[keras.metrics.categorical_accuracy,
                                    tfa.metrics.F1Score(num_classes=6, average='weighted')]
                           )

    def create_layers(self, inputs1, inputs2, inputs3):
        regularizer = None
        dropout_rate = 0.2
        INTERMEDIATE_DIM = 300
        NUM_HEADS = 2
        # create vectorize layer, to transform words in integer
        x = keras_nlp.tokenizers.WordPieceTokenizer('word_piece_vocabulary', 300, lowercase=True,
                                                    strip_accents=True)(inputs1)

        # create embedding layer
        x = keras_nlp.layers.TokenAndPositionEmbedding(76885, 300, 300)(x)

        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(inputs=x)
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(inputs=x)
        x = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(inputs=x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.1)(x)
        outputs = keras.layers.Dense(6, activation="sigmoid")(x)

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
