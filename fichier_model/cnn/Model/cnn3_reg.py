import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa

class Model:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.name = "cnn3_reg"
        inputs1 = keras.Input(shape=(1,), dtype=tf.string)  # text
        inputs2 = keras.Input(shape=1, dtype=tf.float32)  # n_comment
        inputs3 = keras.Input(shape=1, dtype=tf.float32)  # n_votes

        layers = self.create_layers(inputs1, inputs2, inputs3)

        self.model = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=layers)

        self.model.compile(optimizer=keras.optimizers.Adamax(),
                           loss=keras.losses.mse,
                           metrics=[keras.metrics.SparseCategoricalAccuracy(), tfa.metrics.F1Score(num_classes=6, average='weighted')]
                           )

    def create_layers(self, inputs1, inputs2, inputs3):
        regularizer = keras.regularizers.l2(0.00008)
        dropout_rate = 0.05
        # create vectorize layer, to transform words in integer
        vectorize_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            ngrams=2,
            output_sequence_length=1400,
            vocabulary=self.vocabulary
        )(inputs1)

        x = keras.layers.Embedding(len(self.vocabulary) + 1, 32, batch_size=100, embeddings_regularizer=keras.regularizers.l2(0.0008))(
            vectorize_layer)

        x = keras.layers.Conv1D(64, 9, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.MaxPooling1D(pool_size=5)(x)

        x = keras.layers.Conv1D(32, 9, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.MaxPooling1D(pool_size=5)(x)

        x = keras.layers.Conv1D(32, 6, kernel_regularizer=regularizer, bias_regularizer=regularizer, padding="same",
                                activation=keras.activations.relu)(x)
        x = keras.layers.MaxPooling1D(pool_size=10)(x)

        x = keras.layers.LocallyConnected1D(16, 3, kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                            padding="valid", activation=keras.activations.relu)(x)

        x = keras.layers.BatchNormalization()(x)

        flatten = keras.layers.Flatten()(x)

        layer1 = keras.layers.Dense(32, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(flatten)
        drop1 = keras.layers.Dropout(dropout_rate)(layer1)

        layer2 = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop1)
        drop2 = keras.layers.Dropout(dropout_rate)(layer2)

        layer3 = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop2)
        drop3 = keras.layers.Dropout(dropout_rate)(layer3)

        layer4 = keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)(drop3)
        drop4 = keras.layers.Dropout(dropout_rate)(layer4)

        conc = keras.layers.concatenate([drop4, inputs2, inputs3])

        x = keras.layers.Dense(10, activation=keras.activations.tanh)(conc)

        return keras.layers.Dense(1, activation=keras.activations.linear)(x)

    def run_experiment(self, data, output, epochs=10, batch_size=100, validation_split=0.2, tensorboard_callback=None):

        if tensorboard_callback is None:
            res = self.model.fit(data, output, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split)
        else:
            res = self.model.fit(data, output, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split,
                           callbacks=[tensorboard_callback])
        return res

    def get_class_weights(self, output):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(output), y=output)
        di = {}
        for i in range(len(class_weights)):
            di[i] = class_weights[i]
        return di

    def evaluate(self):
        pass
