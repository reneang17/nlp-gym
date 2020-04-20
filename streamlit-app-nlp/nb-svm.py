import streamlit as st
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import numpy as np
import itertools
import matplotlib.pyplot as plt


st.title("Sentiment Analyzer Based On Text Analysis ")
st.subheader("Paras Patidar - MLAIT")
st.write('\n\n')



@st.cache
def get_all_data():
    train = pd.read_csv('./data/train.csv')
    #test = pd.read_csv('./data/test.csv')
    #subm = pd.read_csv('./data/sample_submission.csv')
    return train

train = get_all_data()

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)

if st.checkbox('Show trainig data (head)'):
    st.write(train.head())

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
    
@st.cache
def preprocessing_data(train):
    
    n = train.shape[0]
    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                   min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                   smooth_idf=1, sublinear_tf=1 )
    trn_term_doc = vec.fit_transform(train[COMMENT])
    #test_term_doc = vec.transform(test[COMMENT])

    return trn_term_doc

if st.checkbox('Show preprocessed training data (head)'):
    st.write(preprocessing_data(train))


def training_step(data,vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]
    training_text = vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text,training_result)
