import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from time import time



train = pd.read_csv("../dataset/goodreads_train.csv")



train_prepro = pd.DataFrame(data=np.load(file="../vocabulaires/prepro_train_archive_PN_less.npy", allow_pickle=True), columns=['review_text'])['review_text']


train['review_text'] = train_prepro

rating = keras.utils.to_categorical(train['rating'], num_classes=6)

inputs = keras.Input(shape=(1,), dtype=tf.string)
inputs2 = keras.Input(shape=(1), dtype=tf.int64)
inputs3 = keras.Input(shape=(1), dtype=tf.int64)

vectorize_layer = keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    output_mode='int',
    output_sequence_length=1400,
    vocabulary=np.load('../vocabulaires/voc_lemm_without_NP.npy')
)(inputs)
conc = keras.layers.concatenate([vectorize_layer, inputs2,inputs3])

layer1 = keras.layers.Dense(20, activation=tf.keras.activations.tanh)(conc)
layer2 = keras.layers.Dense(30, activation=tf.keras.activations.tanh)(layer1)
layer3 = keras.layers.Dense(20, activation=tf.keras.activations.tanh)(layer2)

outputs = keras.layers.Dense(6, activation=tf.keras.activations.tanh)(layer3)



model = keras.Model(inputs=[inputs, inputs2, inputs3], outputs=outputs, name="mlp1")

tensorboard = TensorBoard(log_dir="logs/".format(time()))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
              )
model.save("mlp1")
model.fit([train['review_text'], train['n_comments'], train['n_votes']], rating, epochs=10,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(log_dir="../logs/relu"),
                  ],
                  batch_size=100000
                  )

model.save("mlp1_train")

test = pd.read_csv("../dataset/goodreads_test.csv")

test_prepro = pd.DataFrame(data=np.load(file="../vocabulaires/prepro_test_archive_PN_less.npy", allow_pickle=True), columns=['review_text'])['review_text']

res = model.predict([train['review_text'], train['n_comments'], train['n_votes']])

restest = model.predict([test['review_text'], test['n_comments'], test['n_votes']])



ff = []
for line in tqdm(restest):
    tmp = -2
    category = None
    for i in (range(6)):
        if line[i] > tmp:
            category = i
            tmp = line[i]
    ff.append(category)
data = np.array(ff)


test['rating'] = data

id = test['review_id'].to_numpy()
rating = test['rating'].to_numpy()
df = pd.DataFrame( columns=['review_id', 'rating'])

df['review_id'] = id
df['rating'] = rating

df.to_csv('submission_mlp1_model.csv',index=False )