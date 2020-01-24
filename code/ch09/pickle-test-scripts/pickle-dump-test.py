import pickle
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from html.parser import HTMLParser
import shivatokenizer as st

stop = stopwords.words('english')

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=st.mytokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)


df = pd.read_csv('./movie_data_small.csv', encoding='utf-8')

#df.loc[:100, :].to_csv('./movie_data_small.csv', index=None)


X_train = df['review'].values
y_train = df['sentiment'].values

X_train = vect.transform(X_train)
clf.fit(X_train, y_train)

pickle.dump(stop,
            open('stopwords.pkl', 'wb'),
            protocol=4)

pickle.dump(clf,
            open('classifier.pkl', 'wb'),
            protocol=4)
