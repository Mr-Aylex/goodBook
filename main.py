import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import re
import shutil
import string

from py_file.estimate_batch_size import *

train = pd.read_csv("dataset/goodreads_train.csv")
# test = pd.read_csv("dataset/goodreads_test.csv")

x_train = train['review_text']
# x_test = test['review_text']

y_train = train['rating']
# y_test = test['rating']

# layer = tf.keras.layers.StringLookup()
# layer.adapt(data)
# layer.get_vocabulary()


vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens= 100000,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    output_mode='int',
    output_sequence_length=1400,
    vocabulary=np.load('vocabulary.npy'))

print("adapt")
# vectorize_layer.adapt(x_train)
print("model_adding")
print(len(vectorize_layer.get_vocabulary()))


model = tf.keras.models.Sequential()


model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.mse
              )

optimal_batch_size = FindBatchSize(model)

print("training")
model.fit(x_train, y_train, epochs=50,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(log_dir="/logs"),
                  ],
                  batch_size=optimal_batch_size
                  )

model.save("models_trained/linear_model_1")