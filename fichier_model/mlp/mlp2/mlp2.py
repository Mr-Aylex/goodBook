import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from time import time


print("open dataset")
train = pd.read_csv("../dataset/goodreads_train.csv")
#%%
train_prepro = pd.DataFrame(data=np.load(file = "C:/Users/alhus/PycharmProjects/goodBook/prepro_train_archive_PN_less.npy", allow_pickle=True), columns=['review_text'])['review_text']
# test = pd.read_csv("dataset/goodreads_test.csv")
#%%
train_prepro
#%%
train['review_text'] = train_prepro
#%%
train['review_text']
#%%
print("0  ",train[train['rating'] == 0]["rating"].count())
print("1  ", train[train['rating'] == 1]["rating"].count())
print("2  ", train[train['rating'] == 2]["rating"].count())
print("3  ", train[train['rating'] == 3]["rating"].count())
print("4  ", train[train['rating'] == 4]["rating"].count())
print("5  ", train[train['rating'] == 5]["rating"].count())
#train_3_5 = train[train['rating'] >= 3]
#train['rating'] = train['rating'].apply(lambda x: x-3)
#%%
#train = train[train['rating'] >= 3]
#%%
rating = keras.utils.to_categorical(train['rating'], num_classes=6)
#%%
rating = rating.astype(int)
#%%
type(rating)
#%%
mylen = len(np.load('C:/Users/alhus/PycharmProjects/goodBook/voc_lemm_without_NP.npy'))
#%%
mylen
#%%
inputs = keras.Input(shape=(1,), dtype=tf.string)
"""inputs2 = keras.Input(shape=(1), dtype=tf.int64)
inputs3 = keras.Input(shape=(1), dtype=tf.int64)"""

vectorize_layer = keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    output_mode='int',
    output_sequence_length=1400,
    vocabulary=np.load('C:/Users/alhus/PycharmProjects/goodBook/voc_lemm_without_NP.npy')
)(inputs)
#%%
vectorize_layer.shape
#%%
#conc = keras.layers.concatenate([vectorize_layer])

embedding_layer = tf.keras.layers.Embedding(input_dim=mylen+1,output_dim=64,mask_zero=True)(vectorize_layer)
flattened_layer = keras.layers.Flatten()(embedding_layer)
layer1 = keras.layers.Dense(64, activation=tf.keras.activations.tanh)(flattened_layer)
layer2 = keras.layers.Dense(32, activation=tf.keras.activations.tanh)(layer1)
layer3 = keras.layers.Dense(32, activation=tf.keras.activations.tanh)(layer2)
layer4 = keras.layers.Dense(16, activation=tf.keras.activations.tanh)(layer3)


outputs = keras.layers.Dense(6, activation=tf.keras.activations.softmax)(layer4)
#%%
outputs.shape
#%%
model = keras.Model(inputs=[inputs], outputs=outputs, name="mnist_model")
tensorboard = TensorBoard(log_dir="../logs/relu".format(time()))
#%%
#learning_rates = [0.01, 0.001, 0.000001, 0.0000001]
#%%
#for learning_rate in learning_rates:
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
              )
#%%
x_val = train['review_text'][-10000:]
y_val = rating[-10000:]
x_train = train['review_text'][:-10000]
y_train = rating[:-10000]
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train['rating']), y = train['rating'])
#%%
weight = {i : class_weights[i] for i in range(6)}
#%%
weight
#%%
model.fit(train['review_text'], rating, epochs=10,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(log_dir="../logs/relu"),
                  ],
                  batch_size=1000, validation_data=[x_val, y_val], shuffle= True, class_weight= weight
                  )
#%%
model.save("../models_trained/PMC_embedding_model_10_class_weights")
#%%
test = pd.read_csv("../dataset/goodreads_test.csv")
#%%
test_prepro = pd.DataFrame(data=np.load(file="C:/Users/alhus/PycharmProjects/goodBook/prepro_test_archive_PN_less.npy", allow_pickle=True), columns=['review_text'])['review_text']
test['review_text'] = test_prepro
#%%
test
#%%
restest = model.predict([test['review_text']])
#%%
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
#%%
test['rating'] = data
#%%
id = test['review_id'].to_numpy()
rating = test['rating'].to_numpy()
df = pd.DataFrame( columns=['review_id', 'rating'])
#%%
df['review_id'] = id
df['rating'] = rating
#%%
df.to_csv('submission_pmc10_embedding_class_weights_model.csv',index=False )