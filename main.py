import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from nltk.corpus import treebank
# %%
STOP_WORD_MAP = {}
all_stop_word = stopwords.words("english")
for stop_word in all_stop_word:
    STOP_WORD_MAP[stop_word] = stop_word
for name in names.words():
    STOP_WORD_MAP[name.lower()] = name

def parallelize_dataframe(df, func, n_cores=4):
    print("split")

    df_split = np.array_split(df, n_cores)
    del df

    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))

    return df


def prepro_stem(text):
    words = text.lower()
    tokens = nltk.word_tokenize(words)
    words_stop_less = [w for w in tokens if STOP_WORD_MAP.get(w) == None]

    stemmed = [PorterStemmer().stem(w) for w in words_stop_less]
    return " ".join(stemmed)


def prepro_lem(text):
    words = text
    tokens = nltk.word_tokenize(words)
    words_stop_less = [w for w in tokens if STOP_WORD_MAP.get(w) == None]

    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_stop_less]
    tagged = nltk.pos_tag(lemmed)
    sentence = [w[0] for w in tagged if w[1] != 'NNP']

    return " ".join(sentence).lower()

def prepro_lem_neg(text):
    words = text
    tokens = nltk.word_tokenize(words)
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    tokens = mark_negation(lemmed)
    words_stop_less = [w for w in tokens if STOP_WORD_MAP.get(w.lower()) == None]

    tagged = nltk.pos_tag(words_stop_less)
    sentence = [w[0] for w in tagged if w[1] != 'NNP']

    return " ".join(sentence).lower()

def prepro_map(data_frame):
    tqdm.pandas()
    return data_frame.progress_apply(lambda x: prepro_lem_neg(x))


if __name__ == '__main__':
    # print("start preprocessing")
    # print("open dataset")
    train = pd.read_csv("dataset/goodreads_train.csv")
    x_train = train['review_text']
    df_train = parallelize_dataframe(x_train, prepro_map, 15)
    np.save("vocabulaires/prepro_train_archive_NEG_lem", df_train.to_numpy())

    del x_train
    del train
    test = pd.read_csv("dataset/goodreads_test.csv")
    x_test = test['review_text']
    df_test = parallelize_dataframe(x_test, prepro_map, 15)
    np.save("vocabulaires/prepro_test_archive_NEG_lem", df_test.to_numpy())
