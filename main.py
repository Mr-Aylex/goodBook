import numpy as np
import tensorflow as tf
from tensorflow import keras
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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df


def parallelize2_dataframe(df, func, n_cores=4):
    print("split")

    df_split = np.array_split(df, n_cores)
    print("split done")
    with Pool(n_cores) as pool:
        pd.concat(pool.map(func, df_split))
    return 0


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
    return data_frame.progress_apply(lambda x: prepro_stem(x))


def count_numbers_of_each_words(data_frame):
    print("counting")
    print(data_frame.shape)
    f = data_frame['review_text'].str.split(expand=True).stack().value_counts()
    return f

if __name__ == '__main__':
    # get the words from the dataset when the number count is more than 5
    # print("get the words from the dataset when the number count is more than 5")

    train = pd.read_csv("dataset/goodreads_train.csv")
    #set lower case
    # train['review_text'] = train['review_text'].str.lower()
    word_count = train['review_text'].str.lower().str.replace('[^\w\s]','').str.split(expand=True).stack().value_counts()

    # word_count = train['review_text'].str.split(expand=True).stack().value_counts()
    print('word_count')
    word_count = word_count[word_count > 5]
    print('word_count > 5', word_count)
    word_count = word_count.index.tolist()
    word_count = set(word_count)
    print("get the words from the dataset when the number count is more than 5")
    print(word_count)

    # x_train = train['review_text']
    # print(x_train)
    # df_train = parallelize_dataframe(x_train, prepro_map, 15)
    # print(df_train)
    # np.save("dataset/archive/prepro_train_archive_stem", df_train.to_numpy())
    #
    # del x_train
    # del train
    # test = pd.read_csv("dataset/goodreads_test.csv")
    # x_test = test['review_text']
    # print(x_test)
    # df_test = parallelize_dataframe(x_test, prepro_map, 15)
    # print(df_train)
    # np.save("dataset/archive/prepro_test_archive_stem", df_test.to_numpy())
