from sklearn.model_selection import GridSearchCV

from assignment2.alice.features.sites import *
from assignment2.alice.features.time import *
from assignment2.alice.utils import *
#from assignment2.alice.lght import *
from assignment2.alice.sdg import *
import eli5
from bayes_opt import BayesianOptimization
import sys
import warnings

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


def f_tfidf_ngram(range):
    return lambda tr, tst: f_tfidf_sites(tr, tst, max_features=tfidf_feat, ngram_range=(1, int(range)))


features = [f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique,
            f_month, f_year_month, f_weekday,
            f_morning, f_day, f_evening, f_night,
            f_end_morning, f_end_day, f_end_evening, f_end_night,
            f_duration_pow_2, f_year_month_pow_2, f_seconds_per_site_pow_2]
scale_features = [f_month, f_year_month, f_weekday, f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique,
                  f_duration_pow_2, f_year_month_pow_2, f_seconds_per_site_pow_2]

basic_features = [f_tfidf_ngram(4), f_duration, f_unique,
                     f_year_month, f_weekday,
                     f_morning, f_day, f_evening,
                     f_end_morning, f_end_evening,
                     f_year_month_pow_2]
opt_args = get_est_params()
# opt_args = {f.__name__: (-1, 1) for f in features if f not in basic_features}

f_dict = {f.__name__: f for f in features}



def solve():
    train_df, test_df = load_data()
    y_train = train_df['target'].astype('int').values

    def score(**kwargs):
        features_to_use = basic_features
        #for key, value in kwargs.items():
        #    if key.startswith('f_'):
        #        if value >= 0:
        #            features_to_use.append(f_dict.get(key))
        return score_and_write(features_to_use, kwargs)

    def score_and_write(features_to_use, kwargs, C=2.85, file_postfix=""):
        X_train, X_test, f_names = build_features(train_df, test_df,
                                                  features_to_use,
                                                  scale_features=scale_features)
        score, est = cross_validate(X_train, y_train, kwargs, C=C)
        prediction = make_prediction(est, X_train, y_train, X_test)

        if True:
            eli5.show_weights(estimator=est,
                              feature_names=f_names, top=30)
            print('Score:', str(score))
            print('New feature weights:')

            print(pd.DataFrame({'feature': f_names[tfidf_feat:],
                                'coef': est.coef_.flatten()[-len(f_names[tfidf_feat:]):]}))
        write_to_submission_file(prediction, str(score) + "_" + file_postfix + '.csv')
        return score

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

    #gridsearch()
    score_and_write(basic_features, {}, file_postfix="tuned_c")
    #score_and_write([f_tfidf_ngram(4), f_duration, f_unique,
    #                 f_year_month, f_weekday,
    #                 f_morning, f_day, f_evening,
    #                 f_end_morning, f_end_evening,
    #                 f_year_month_pow_2], file_postfix="all_neg")


if __name__ == "__main__":
    solve()
