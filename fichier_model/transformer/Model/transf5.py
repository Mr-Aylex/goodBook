import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa

class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim), ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

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
        self.name = "transf4"
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
        regularizer = keras.regularizers.l2(0.000008)
        dropout_rate = 0.2
        # create vectorize layer, to transform words in integer
        vectorize_layer = keras.layers.TextVectorization(
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            output_mode='int',
            output_sequence_length=192,
            vocabulary=self.vocabulary
        )(inputs1)
        """
        x = keras.layers.Embedding(672022, 32, batch_size=100, embeddings_regularizer=keras.regularizers.l2(0.0008))(
            vectorize_layer)"""
        embedding_layer = TokenAndPositionEmbedding(192, len(self.vocabulary), 300)  # 672022
        x = embedding_layer(vectorize_layer)

        transformer_block = TransformerBlock(300, 100, 40, dropout_rate=dropout_rate)  # embed_dim, num_heads, ff_dim

        x = transformer_block(x)

        #x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Conv1D(128, 3, activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)


        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(16, activation="relu")(x)

        # conc = keras.layers.concatenate([x, inputs2, inputs3])

        return keras.layers.Dense(6, activation=keras.activations.sigmoid)(x)

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
                                 callbacks=[callbacks])
        return res

    def get_class_weights(self, output):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(output), y=output)
        di = {}
        for i in range(len(class_weights)):
            di[i] = class_weights[i]
        return di

    def evaluate(self):
        pass
