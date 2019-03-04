import numpy as np

from assignment2.alice.features.sites import f_unique

times = ['time%s' % i for i in range(1, 11)]
hour = lambda df: df['time1'].apply(lambda ts: ts.hour)
morning = lambda h: ((h >= 7) & (h <= 11)).astype('int').values.reshape(-1, 1)
day = lambda h: ((h >= 12) & (h <= 18)).astype('int').values.reshape(-1, 1)
evening = lambda h: ((h >= 19) & (h <= 23)).astype('int').values.reshape(-1, 1)
night = lambda h: ((h >= 0) & (h <= 6)).astype('int').values.reshape(-1, 1)

end_hour = lambda df: df[times].max(axis=1).apply(lambda d: d.hour)
seconds = lambda df: ((df[times].max(axis=1) - df[times].min(axis=1)) / np.timedelta64(1, 's'))

rp = lambda df: df.values.reshape(-1, 1)


def f_seconds(train_df, test_df):
    return rp(seconds(train_df)), rp(seconds(test_df))


def f_seconds_per_site(train_df, test_df):
    def divide(a, b):
        return (a.flatten() / b.flatten()).reshape(-1, 1)

    f_sec_tr, f_sec_test = f_seconds(train_df, test_df)
    f_uni_tr, f_uni_test = f_unique(train_df, test_df)

    return divide(f_sec_tr, f_uni_tr), divide(f_sec_test, f_uni_test)


def f_hour(train_df, test_df):
    return rp(hour(train_df)), rp(hour(test_df))


def f_morning(train_df, test_df):
    return morning(hour(train_df)), morning(hour(test_df))


def f_day(train_df, test_df):
    return day(hour(train_df)), day(hour(test_df))


def f_evening(train_df, test_df):
    return evening(hour(train_df)), evening(hour(test_df))


def f_night(train_df, test_df):
    return night(hour(train_df)), night(hour(test_df))


def f_end_hour(train_df, test_df):
    return rp(end_hour(train_df)), rp(end_hour(test_df))


def f_end_morning(train_df, test_df):
    return morning(end_hour(train_df)), morning(end_hour(test_df))


def f_end_day(train_df, test_df):
    return day(end_hour(train_df)), day(end_hour(test_df))


def f_end_evening(train_df, test_df):
    return evening(end_hour(train_df)), evening(end_hour(test_df))


def f_end_night(train_df, test_df):
    return night(end_hour(train_df)), night(end_hour(test_df))
