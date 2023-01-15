import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa

class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Model:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.name = "cnn15"
        inputs1 = keras.Input(shape=(1,), dtype=tf.string)  # text
        inputs2 = keras.Input(shape=1, dtype=tf.float32)  # n_comment
        inputs3 = keras.Input(shape=1, dtype=tf.float32)  # n_votes

        layers = self.create_layers(inputs1, inputs2, inputs3)

        self.model = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=layers)

        self.model.compile(optimizer=keras.optimizers.Adamax(),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=[keras.metrics.categorical_accuracy, tfa.metrics.F1Score(num_classes=6, average='weighted')]
                           )

    def create_layers(self, inputs1, inputs2, inputs3):
        regularizer = keras.regularizers.l2(0.000001)
        dropout_rate = 0.15
        # create vectorize layer, to transform words in integer
        vectorize_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=200,
            vocabulary=self.vocabulary
        )(inputs1)

        x = TokenAndPositionEmbedding(200, len(self.vocabulary), 200)(vectorize_layer)

        x = keras.layers.Conv1D(128, 9, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)

        x = keras.layers.Conv1D(64, 9, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = keras.layers.MaxPooling1D(pool_size=2)(x)

        x = keras.layers.Conv1D(32, 6, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)

        x = keras.layers.BatchNormalization()(x)

        flatten = keras.layers.Flatten()(x)

        layer1 = keras.layers.Dense(64, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(flatten)
        drop1 = keras.layers.Dropout(dropout_rate)(layer1)

        layer2 = keras.layers.Dense(32, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop1)
        drop2 = keras.layers.Dropout(dropout_rate)(layer2)

        layer3 = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop2)
        drop3 = keras.layers.Dropout(dropout_rate)(layer3)

        layer4 = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop3)
        drop4 = keras.layers.Dropout(dropout_rate)(layer4)

        input_dense2 = keras.layers.Dense(8, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                          bias_regularizer=regularizer)(inputs2)
        input_dense3 = keras.layers.Dense(8, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                          bias_regularizer=regularizer)(inputs3)

        conc = keras.layers.concatenate([drop4, input_dense2, input_dense3])
        drop = keras.layers.Dropout(dropout_rate)(conc)

        conc_dense = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer)(drop)

        drop = keras.layers.Dropout(dropout_rate)(conc_dense)

        return keras.layers.Dense(6, activation=keras.activations.sigmoid)(drop)

    def run_experiment(self, data, output, epochs=10, batch_size=100, validation_split=0.2, callbacks=None):

        rating = keras.utils.to_categorical(output, num_classes=6)

        class_weight = self.get_class_weights(output)
        if callbacks is None:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split#, class_weight=class_weight
                                 )
        else:
            res = self.model.fit(data, rating, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split,# class_weight=class_weight,
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
