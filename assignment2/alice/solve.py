from sklearn.model_selection import GridSearchCV

from assignment2.alice.features.sites import *
from assignment2.alice.features.time import *
from assignment2.alice.utils import *
# from assignment2.alice.lght import *
from assignment2.alice.sdg import *
import eli5
from bayes_opt import BayesianOptimization
import sys
import warnings
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

if not sys.warnoptions:
    warnings.simplefilter("ignore")

tfidf_feat = 50000

blend = [
    ["0.9257123911573547_all_neg.csv", 0.8],
    ["0.9100756154674823_.csv", 0.8],
]


def blend():
    df1 = pd.read_csv('output/0.9257123911573547_all_neg.csv')
    df2 = pd.read_csv('output/0.9100756154674823_.csv')
    df1['target'] = 0.95 * df1['target'] + 0.05 * df2['target']
    df1.to_csv('output/bl_sub.csv', index=False)


def f_sites_ngram(range):
    return lambda tr, tst: f_sites(tr, tst, ngram_range=(1, int(range)))


def f_tfidf_ngram(range, feat_num=tfidf_feat):
    return lambda tr, tst: f_tfidf_sites(tr, tst, max_features=feat_num, ngram_range=(1, int(range)))


features = [f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique,
            f_month, f_year_month, f_weekday,
            f_morning, f_day, f_evening, f_night,
            f_end_morning, f_end_day, f_end_evening, f_end_night,
            f_duration_pow_2, f_year_month_pow_2, f_seconds_per_site_pow_2,
            ]
scale_features = [f_month, f_year_month, f_weekday, f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique,
                  f_duration_pow_2, f_year_month_pow_2, f_seconds_per_site_pow_2, extract_unique]

basic_features = [f_tfidf_ngram(4), f_duration, f_unique,
                  f_year_month, f_weekday,
                  f_morning, f_day, f_evening,
                  f_end_morning, f_end_evening,
                  f_year_month_pow_2]

good_features = [wrap_f(f) for f in [
    extract_time_features,
    extract_unique,
    extract_year_month,
    extract_part_of_day,
    extract_weekend,
    extract_duration,
    extract_week
]]

basic_features_plain = [
    f_duration, f_unique,
    f_year_month, f_weekday,
    f_morning, f_day, f_evening,
    f_end_morning, f_end_evening,
    f_year_month_pow_2
]

dummy_features = [wrap_f(f) for f in [
    f_hour_dummies,
    f_weekday_dummies,
    f_month_dummies,
    f_year_dummies

]]
opt_args = get_est_params()
# opt_args = {f.__name__: (-1, 1) for f in features if f not in basic_features}

f_dict = {f.__name__: f for f in features}


def solve():
    train_df, test_df = load_data()
    y_train = train_df['target'].astype('int').values

    def score(**kwargs):
        features_to_use = basic_features
        # for key, value in kwargs.items():
        #    if key.startswith('f_'):
        #        if value >= 0:
        #            features_to_use.append(f_dict.get(key))
        return score_and_write(features_to_use, kwargs)

    def score_and_write(est, features_to_use, kwargs, C=2.85, file_postfix=""):
        X_train, X_test = build_features(train_df, test_df,
                                         features_to_use,
                                         scale_features=scale_features)
        sfs1 = SFS(est,
                   forward=True,
                   floating=False,
                   verbose=2,
                   scoring='roc_auc',
                   cv=0)

        sfs1 = sfs1.fit(X_train.tocsc(), y_train)
        score = cross_validate(est, X_train, y_train)
        # score = score_model(est, X_train, y_train)
        print('Score ' + file_postfix + ' :', str(score))
        prediction = make_prediction(est, X_train, y_train, X_test)

        write_to_submission_file(prediction, str(np.mean(score)) + "_" + file_postfix + '.csv')
        return np.mean(score)

    def gridsearch():
        X_train, X_test, f_names = build_features(train_df, test_df,
                                                  basic_features,
                                                  scale_features=scale_features)
        est = LogisticRegression(C=3, random_state=17, solver='liblinear')
        params = {
            "C": np.linspace(2.7, 3, num=25)
        }
        time_split = TimeSeriesSplit(n_splits=10)
        logit_grid_searcher = GridSearchCV(estimator=est, param_grid=params, scoring='roc_auc', n_jobs=1,
                                           cv=time_split, verbose=10)

        logit_grid_searcher.fit(X_train, y_train)
        print(logit_grid_searcher.best_params_)

    def opt():
        optimizer = BayesianOptimization(
            f=score,
            pbounds=opt_args,
            random_state=1,
        )
        optimizer.maximize(
            init_points=3,
            n_iter=15,
            # What follows are GP regressor parameters
            alpha=1e-3,
            n_restarts_optimizer=5
        )

    # gridsearch()
    score_and_write(LogisticRegression(C=21.5, random_state=17, solver='liblinear'),
                    basic_features + dummy_features, {}, file_postfix="dummy_features")
    # score_and_write(LogisticRegression(C=21.5, random_state=17, solver='liblinear'),
    #                [f_tfidf_ngram(4, feat_num=50000)] + good_features_2, {}, file_postfix="good_features_2")


if __name__ == "__main__":
    solve()
