import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import re
import shutil
import string
from gensim.parsing.preprocessing import remove_stopwords
from py_file.estimate_batch_size import *

from py_file.estimate_batch_size import *

train = pd.read_csv("dataset/goodreads_train.csv")
# test = pd.read_csv("dataset/goodreads_test.csv")

x_train = train['review_text']
# x_test = test['review_text']

x_train_rm_st = x_train

y_train = train['rating']
# y_test = test['rating']
print("cleaning data")


x_train_rm_st = x_train_rm_st.apply(lambda x: remove_stopwords(x))

#for x in range(len(x_train)):

    #x_train_rm_st.iloc[x] = remove_stopwords(x_train.iloc[x])


print(len(x_train_rm_st.iloc[0]))

print("tokenizing data")
layer = tf.keras.layers.StringLookup()
layer.adapt(x_train_rm_st)
print(type(layer.get_vocabulary()))



"""vectorize_layer = tf.keras.layers.TextVectorization(
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
# model.add(tf.keras.layers.Dense(255, activation=tf.keras.activations.softmax))
# model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.softmax))
# model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.softmax))
# model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.softmax))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.KLD
              )

optimal_batch_size = FindBatchSize(model)


print("training")
model.summary()
model.fit(x_train, y_train, epochs=5,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(log_dir="/logs"),
                  ],
                  batch_size=150000
                  )

model.save("models_trained/linear_model_1")"""