import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from multiprocessing import  Pool
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank

#%%
dic = {}
all_stop_word = stopwords.words("english")
for stop_word in all_stop_word:
    dic[stop_word] = stop_word
def parallelize_dataframe(df, func, n_cores=4):
    print("split")

    df_split = np.array_split(df, n_cores)
    del df
    print(df_split)

    with Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, df_split))

    return df
def prepro(text):
    words = text.lower()
    tokens = nltk.word_tokenize(words)
    words_stop_less = [w for w in tokens if dic.get(w) == None]
    
    stemmed = [PorterStemmer().stem(w) for w in words_stop_less]
    return " ".join(stemmed)


def prepro2(text):
    words = text
    tokens = nltk.word_tokenize(words)
    words_stop_less = [w for w in tokens if dic.get(w) == None]

    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_stop_less]
    tagged = nltk.pos_tag(lemmed)
    sentence = [w[0] for w in tagged if w[1] != 'NNP']



    return " ".join(sentence).lower()

def prepro_map(data_frame):
    tqdm.pandas()
    return data_frame.progress_apply(lambda x: prepro2(x))
#%%
print("start preprosessing")
if __name__ ==  '__main__':
    print("open dataset")
    train = pd.read_csv("dataset/goodreads_train.csv")

    # test = pd.read_csv("dataset/goodreads_test.csv")
    x_train = train['review_text']
    df = parallelize_dataframe(x_train, prepro_map, 15)



    np.save("prepro_train_archive_PN_less",df.to_numpy())
