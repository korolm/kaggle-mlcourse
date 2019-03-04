from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

sites = ['site%s' % i for i in range(1, 11)]


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


def f_sites(train_df, test_df, ngram_range=(1, 3)):
    train_file, test_file = transform_to_txt_format(train_df, test_df)
    cv = CountVectorizer(ngram_range=ngram_range, max_features=50000)
    with open(train_file) as inp_train_file:
        X_train = cv.fit_transform(inp_train_file)
    with open(test_file) as inp_test_file:
        X_test = cv.transform(inp_test_file)
    return X_train, X_test


def count_not_zeros(x):
    unique = set(x)
    if 0 in unique:
        unique.discard(0)
    return len(unique)


unique_sites = lambda df: np.array([count_not_zeros(x) for x in df[sites].values]).reshape(-1, 1)


def f_unique(traim_df, test_df):
    return unique_sites(traim_df), unique_sites(test_df)
