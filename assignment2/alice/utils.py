import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from assignment2.alice.sdg import *

times = ['time%s' % i for i in range(1, 11)]


def load_data():
    # %%
    train_df = pd.read_csv('data/train_sessions.csv',
                           index_col='session_id', parse_dates=times)
    test_df = pd.read_csv('data/test_sessions.csv',
                          index_col='session_id', parse_dates=times)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')
    return train_df, test_df


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv('output/' + out_file, index_label=index_label)


def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def build_features(train_df, test_df, features, scale_features = []):
    X_train_a = []
    X_test_a = []
    f_names = []
    for f in features:
        X_train_f, X_test_f,f_name = f(train_df, test_df)
        if f in scale_features:
            X_train_f, X_test_f = scale(X_train_f, X_test_f)
        X_train_a.append(X_train_f)
        X_test_a.append(X_test_f)
        f_names.append(f_name)
    return sparse.hstack(X_train_a), sparse.hstack(X_test_a), np.hstack(f_names)


def cross_validate(X_train, y_train, kwargs, C=2.85):
    #est = create_est(kwargs)
    est = LogisticRegression(C=C, random_state=17, solver='liblinear')
    time_split = TimeSeriesSplit(n_splits=10)
    cv_scores = cross_val_score(est, X_train, y_train, cv=time_split,
                                scoring='roc_auc', n_jobs=1)  # hangs with n_jobs > 1, and locally this runs much faster
    return cv_scores.mean(), est


def make_prediction(est, X_train, y_train, X_test):
    est.fit(X_train, y_train)
    return est.predict_proba(X_test)[:, 1]
