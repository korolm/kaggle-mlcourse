import numpy as np
from assignment2.alice.features.sites import f_unique
from joblib import Memory

cachedir = 'cache/'
memory = Memory(cachedir, verbose=0)

times = ['time%s' % i for i in range(1, 11)]
hour = lambda df: df['time1'].apply(lambda ts: ts.hour)
weekday = lambda df: df['time1'].apply(lambda ts: ts.weekday())
month = lambda df: df['time1'].apply(lambda ts: ts.month)
year_month = lambda df: df['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5
morning = lambda h: ((h >= 7) & (h <= 11)).astype('int').values.reshape(-1, 1)
day = lambda h: ((h >= 12) & (h <= 18)).astype('int').values.reshape(-1, 1)
evening = lambda h: ((h >= 19) & (h <= 23)).astype('int').values.reshape(-1, 1)
night = lambda h: ((h >= 0) & (h <= 6)).astype('int').values.reshape(-1, 1)

end_hour = lambda df: df[times].max(axis=1).apply(lambda d: d.hour)
duration = lambda df: ((df[times].max(axis=1) - df[times].min(axis=1)).astype('timedelta64[ms]').astype(int))
rp = lambda df: df.values.reshape(-1, 1)


@memory.cache
def f_duration(train_df, test_df):
    return rp(duration(train_df)), rp(duration(test_df)), ['duration']


@memory.cache
def f_seconds_per_site(train_df, test_df):
    def divide(a, b):
        return (a.flatten() / b.flatten()).reshape(-1, 1)

    f_sec_tr, f_sec_test, f_name_1 = f_duration(train_df, test_df)
    f_uni_tr, f_uni_test, f_name_1 = f_unique(train_df, test_df)

    return divide(f_sec_tr, f_uni_tr), divide(f_sec_test, f_uni_test), ['seconds_per_site']


@memory.cache
def f_hour(train_df, test_df):
    return rp(hour(train_df)), rp(hour(test_df)), ['hour']


@memory.cache
def f_weekday(train_df, test_df):
    return rp(weekday(train_df)), rp(weekday(test_df)), ['weekday']


@memory.cache
def f_month(train_df, test_df):
    return rp(month(train_df)), rp(month(test_df)), ['month']


@memory.cache
def f_year_month(train_df, test_df):
    return (year_month(train_df)), (year_month(test_df)), ['year_month']


@memory.cache
def f_morning(train_df, test_df):
    return morning(hour(train_df)), morning(hour(test_df)), ['morning']


@memory.cache
def f_day(train_df, test_df):
    return day(hour(train_df)), day(hour(test_df)), ['day']


@memory.cache
def f_evening(train_df, test_df):
    return evening(hour(train_df)), evening(hour(test_df)), ['evening']


@memory.cache
def f_night(train_df, test_df):
    return night(hour(train_df)), night(hour(test_df)), ['night']


@memory.cache
def f_end_hour(train_df, test_df):
    return rp(end_hour(train_df)), rp(end_hour(test_df)), ['end_hour']


@memory.cache
def f_end_morning(train_df, test_df):
    return morning(end_hour(train_df)), morning(end_hour(test_df)), ['end_morning']


@memory.cache
def f_end_day(train_df, test_df):
    return day(end_hour(train_df)), day(end_hour(test_df)), ['end_day']


@memory.cache
def f_end_evening(train_df, test_df):
    return evening(end_hour(train_df)), evening(end_hour(test_df)), ['end_evening']


@memory.cache
def f_end_night(train_df, test_df):
    return night(end_hour(train_df)), night(end_hour(test_df)), ['end_night']
