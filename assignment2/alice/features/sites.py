from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from joblib import Memory

cachedir = 'cache/'
memory = Memory(cachedir, verbose=0)

path_to_site_dict = 'data/site_dic.pkl'


def load_site_dict():
    with open(path_to_site_dict, 'rb') as f:
        site2id = pickle.load(f)
    id2site = {v: k for (k, v) in site2id.items()}
    # we treat site with id 0 as "unknown"
    id2site[0] = 'unknown'
    return id2site


sites = ['site%s' % i for i in range(1, 11)]
id2site = load_site_dict()


def transform_to_txt_format(train_df, test_df):
    train_file = 'tmp/train_sessions_text.txt'
    test_file = 'tmp/test_sessions_text.txt'
    sites = ['site%s' % i for i in range(1, 11)]
    train_df[sites].fillna(0).astype('int').to_csv(train_file,
                                                   sep=' ',
                                                   index=None, header=None)
    test_df[sites].fillna(0).astype('int').to_csv(test_file,
                                                  sep=' ',
                                                  index=None, header=None)
    return train_file, test_file


@memory.cache
def f_sites(train_df, test_df, ngram_range=(1, 3)):
    train_file, test_file = transform_to_txt_format(train_df, test_df)
    cv = CountVectorizer(ngram_range=ngram_range, max_features=50000)
    with open(train_file) as inp_train_file:
        X_train = cv.fit_transform(inp_train_file)
    with open(test_file) as inp_test_file:
        X_test = cv.transform(inp_test_file)
    return X_train, X_test#, cv.get_feature_names()


@memory.cache
def f_tfidf_sites(train_df, test_df, ngram_range=(1, 5), sub=False, max_features=50000):
    def join_row(row):
        return ' '.join([id2site[i] for i in row])

    train_sessions = train_df[sites].fillna(0).astype('int').apply(join_row, axis=1)
    test_sessions = test_df[sites].fillna(0).astype('int').apply(join_row, axis=1)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                 max_features=max_features,
                                 tokenizer=lambda s: s.split())
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    return X_train, X_test#, vectorizer.get_feature_names()


@memory.cache
def time_sites(train_df, test_df, ngram_range=(1, 5), max_features=50000):
    time_diff = ['time_diff_%s' % i for i in range(1, 11)]

    def est_session_length(s):
        if s <= 5:
            return 'small'
        if 6 <= s <= 30:
            return 'medium'
        if 31 <= s <= 90:
            return 'large'
        if 91 <= s:
            return 'extra-large'

    def join_row_with_time(row):
        # str_sites = []
        # for i in range(1, 11):
        #    site_id = row['site%s' % i]
        #    if np.isnan(site_id):
        #        site_str = 'no_site'
        #    else:
        #        site_str = str(id2site[row['site%s' % i]])
        #    diff_str = str(row['time_diff_%s' % i])
        #    str_sites.append(site_str + '_' + diff_str)
        return ' '.join(['no_site' + '_' + str(row['time_diff_%s' % i])
                         if np.isnan(row['site%s' % i])
                         else str(id2site[row['site%s' % i]]) + '_' + str(row['time_diff_%s' % i])
                         for i in range(1, 11)])

    for t in range(1, 10):
        train_df['time_diff_' + str(t)] = (
                (train_df['time' + str(t + 1)] - train_df['time' + str(t)]) / np.timedelta64(1, 's')).apply(
            est_session_length)
        test_df['time_diff_' + str(t)] = (
                (test_df['time' + str(t + 1)] - test_df['time' + str(t)]) / np.timedelta64(1, 's')).apply(
            est_session_length)

    train_df['time_diff_10'] = None
    test_df['time_diff_10'] = None

    train_df[sites].fillna(0).astype('int')
    test_df[sites].fillna(0).astype('int')
    train_sessions = train_df[sites + time_diff].apply(join_row_with_time, axis=1)

    test_sessions = test_df[sites + time_diff].apply(join_row_with_time, axis=1)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                 max_features=max_features,
                                 tokenizer=lambda s: s.split())
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    return X_train, X_test#, vectorizer.get_feature_names()


def count_not_zeros(x):
    unique = set(x)
    if 0 in unique:
        unique.discard(0)
    return len(unique)


unique_sites = lambda df: np.array([count_not_zeros(x) for x in df[sites].values]).reshape(-1, 1)


def f_unique(traim_df, test_df):
    return unique_sites(traim_df), unique_sites(test_df), ['unique']

def extract_unique(df):
    data = df[sites].fillna(0).astype('int')
    return csr_matrix([[sum(1 for s in np.unique(row.values) if s != 0)] for _, row in data.iterrows()])
