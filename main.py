import pandas as pd
import tensorflow as tf
import numpy as np
from multiprocessing import  Pool
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import re
import shutil
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from py_file.estimate_batch_size import *

# %%
train = pd.read_csv("dataset/goodreads_train.csv")
# test = pd.read_csv("dataset/goodreads_test.csv")

x_train = train['review_text']


# x_test = test['review_text']
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def prepro(data):
    words = data.lower()
    print("word lower")
    tokens = nltk.word_tokenize(words)
    print("end tokenize")
    words_stop_less = [w for w in tokens if w not in stopwords.words("english")]
    print("end removing stop word")
    stemmed = [PorterStemmer().stem(w) for w in words_stop_less]
    print("end PorterStemmer")
    return " ".join(stemmed)




"""print(x_train[0])

print(len(x_train[0]))
print(prepro(x_train[0]))
print(len(prepro(x_train[0])))"""
x_train_prepro = prepro(x_train[0]).split()

for x, y in x_train[0].split(), x_train_prepro:
    print(x,y)
#parallelize_dataframe(x_train, prepro)
"""x_train = x_train.apply(lambda x: prepro(x))

print(x_train[0])
y_train = train['rating']"""
# y_test = test['rating']
# %%
"""vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=376576,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    output_mode='int',
    output_sequence_length=1400,
    vocabulary=np.load('voc.npy'))
# %%
# print("adapt")
# vectorize_layer.adapt(x_train)
# print("model_adding")
# print(len(vectorize_layer.get_vocabulary()))
# %%
model = tf.keras.models.Sequential()
# %%
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))
# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.mse
              )
# %%
optimal_batch_size = FindBatchSize(model)
print(optimal_batch_size)
# %%
print("training")
# %%
model.fit(x_train, y_train, epochs=50,
          callbacks=[
              tf.keras.callbacks.TensorBoard(log_dir="/logs"),
          ],
          batch_size=100000
          )
# %%
model.save("models_trained/linear_model_1")"""
