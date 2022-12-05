from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

train = pd.read_csv("../dataset/goodreads_train.csv")

train_prepro = pd.DataFrame(data=np.load(file="../vocabulaires/prepro_train_archive_PN_less.npy", allow_pickle=True), columns=['review_text'])['review_text']


train['review_text'] = train_prepro

sia = SentimentIntensityAnalyzer()

res = sia.polarity_scores(train['review_text'][5])


print(res)
print(train['rating'][5])