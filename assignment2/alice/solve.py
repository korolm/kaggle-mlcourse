from assignment2.alice.features.sites import f_sites
from assignment2.alice.features.time import *
from assignment2.alice.utils import *
from bayes_opt import BayesianOptimization

# Bounded region of parameter space

features = [f_hour, f_end_hour, f_seconds, f_seconds_per_site, f_unique,
            f_morning, f_day, f_evening, f_night,
            f_end_morning, f_end_day, f_end_evening, f_end_night]
scale_features = [f_hour, f_end_hour, f_seconds, f_seconds_per_site, f_unique]

opt_args = {f.__name__: (-1, 1) for f in features}
f_dict = {f.__name__: f for f in features}
opt_args['C'] = (0.01, 100)


def solve():
    train_df, test_df = load_data()
    y_train = train_df['target'].astype('int').values

    def score(**kwargs):
        features_to_use = [f_sites]
        for key, value in kwargs.items():
            if key.startswith('f_'):
                if value >= 0:
                    features_to_use.append(f_dict.get(key))
        X_train, X_test = build_features(train_df, test_df,
                                         features_to_use,
                                         scale_features=scale_features)
        score, est = cross_validate(X_train, y_train, C=kwargs.get('C'))
        return score

    optimizer = BayesianOptimization(
        f=score,
        pbounds=opt_args,
        random_state=1,
    )
    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )
    # prediction = make_prediction(est, X_train, y_train, X_test)
    # write_to_submission_file(prediction, 'subm1.csv')


if __name__ == "__main__":
    solve()
