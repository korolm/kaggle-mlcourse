from assignment2.alice.features.sites import *
from assignment2.alice.features.time import *
from assignment2.alice.utils import *
from bayes_opt import BayesianOptimization
import eli5
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

tfidf_feat = 50000


def f_sites_ngram(range):
    return lambda tr, tst: f_sites(tr, tst, ngram_range=(1, int(range)))


def f_tfidf_ngram(range):
    return lambda tr, tst: f_tfidf_sites(tr, tst, max_features=tfidf_feat, ngram_range=(1, int(range)))


features = [f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique,
            f_month, f_year_month, f_weekday,
            f_morning, f_day, f_evening, f_night,
            f_end_morning, f_end_day, f_end_evening, f_end_night]
scale_features = [f_month, f_year_month, f_weekday, f_hour, f_end_hour, f_duration, f_seconds_per_site, f_unique]

basic_features = [f_tfidf_ngram(4), f_morning, f_day, f_evening, f_night, f_duration, f_weekday, f_month, f_year_month]
opt_args = {}
# opt_args = {f.__name__: (-1, 1) for f in features if f not in basic_features}
f_dict = {f.__name__: f for f in features}

opt_args['RS'] = (1, 125)


# opt_args['C'] = (0.01, 0.25)
# opt_args['ngram_range'] = (1, 4)


def solve():
    train_df, test_df = load_data()
    y_train = train_df['target'].astype('int').values

    def score(**kwargs):
        features_to_use = basic_features
        for key, value in kwargs.items():
            if key.startswith('f_'):
                if value >= 0:
                    features_to_use.append(f_dict.get(key))
        return score_and_write(features_to_use)

    def score_and_write(features_to_use, C=3.359, file_postfix=""):
        X_train, X_test, f_names = build_features(train_df, test_df,
                                                  features_to_use,
                                                  scale_features=scale_features)
        score, est = cross_validate(X_train, y_train, C=C)
        prediction = make_prediction(est, X_train, y_train, X_test)
        eli5.show_weights(estimator=est,
                          feature_names=f_names, top=30)

        if True:
            print('New feature weights:')

            print(pd.DataFrame({'feature': f_names[tfidf_feat:],
                                'coef': est.coef_.flatten()[-len(f_names[tfidf_feat:]):]}))
        write_to_submission_file(prediction, str(score) + "_" + file_postfix + '.csv')
        return score

    def opt():
        optimizer = BayesianOptimization(
            f=score,
            pbounds=opt_args,
            random_state=1,
        )
        optimizer.maximize(
            init_points=1,
            n_iter=15,
            # What follows are GP regressor parameters
            alpha=1e-3,
            n_restarts_optimizer=5
        )

    # opt()
    # score_and_write(basic_features)

    basic_features = [f_tfidf_ngram(4)]  # f_morning, f_day, f_evening, f_duration, f_weekday , f_year_month]
    score_and_write([f_tfidf_ngram(4), f_duration, f_unique,
                     f_year_month, f_weekday,
                     f_morning, f_day, f_evening,
                     f_end_morning, f_end_evening], file_postfix="all_neg")


if __name__ == "__main__":
    solve()
