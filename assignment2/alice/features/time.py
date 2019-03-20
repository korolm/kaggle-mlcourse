import numpy as np
from scipy.sparse import csr_matrix

from assignment2.alice.features.sites import f_unique
from joblib import Memory
import pandas as pd
import tqdm

cachedir = 'cache/'
memory = Memory(cachedir, verbose=0)

times = ['time%s' % i for i in range(1, 11)]
hour = lambda df: df['time1'].apply(lambda ts: ts.hour)
weekday = lambda df: df['time1'].apply(lambda ts: ts.weekday())
month = lambda df: df['time1'].apply(lambda ts: ts.month)
year = lambda df: df['time1'].apply(lambda ts: ts.year)
year_month = lambda df: df['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5
morning = lambda h: ((h >= 7) & (h <= 11)).astype('int').values.reshape(-1, 1)
day = lambda h: ((h >= 12) & (h <= 18)).astype('int').values.reshape(-1, 1)
evening = lambda h: ((h >= 19) & (h <= 23)).astype('int').values.reshape(-1, 1)
night = lambda h: ((h >= 0) & (h <= 6)).astype('int').values.reshape(-1, 1)

end_hour = lambda df: df[times].max(axis=1).apply(lambda d: d.hour)
duration = lambda df: ((df[times].max(axis=1) - df[times].min(axis=1)).astype('timedelta64[ms]').astype(int))
rp = lambda df: df.values.reshape(-1, 1)


def divide(a, b):
    return (a.flatten() / b.flatten()).reshape(-1, 1)


def multiply(a, b):
    return (a.flatten() * b.flatten()).reshape(-1, 1)


@memory.cache
def f_duration(train_df, test_df):
    return rp(duration(train_df)), rp(duration(test_df)), ['duration']


@memory.cache
def f_duration_pow_2(train_df, test_df):
    f_sec_tr, f_sec_test, f_name_1 = f_duration(train_df, test_df)
    return multiply(f_sec_tr, f_sec_tr), multiply(f_sec_test, f_sec_test), ['duration_pow_2']


@memory.cache
def f_year_month(train_df, test_df):
    return (year_month(train_df)), (year_month(test_df)), ['year_month']


@memory.cache
def f_year_month_pow_2(train_df, test_df):
    f_sec_tr, f_sec_test, f_name_1 = f_year_month(train_df, test_df)
    return multiply(f_sec_tr, f_sec_tr), multiply(f_sec_test, f_sec_test), ['year_month_pow_2']


@memory.cache
def f_seconds_per_site(train_df, test_df):
    f_sec_tr, f_sec_test, f_name_1 = f_duration(train_df, test_df)
    f_uni_tr, f_uni_test, f_name_1 = f_unique(train_df, test_df)

    return divide(f_sec_tr, f_uni_tr), divide(f_sec_test, f_uni_test), ['seconds_per_site']


@memory.cache
def f_seconds_per_site_pow_2(train_df, test_df):
    f_sec_tr, f_sec_test, f_name_1 = f_seconds_per_site(train_df, test_df)
    return multiply(f_sec_tr, f_sec_tr), multiply(f_sec_test, f_sec_test), ['seconds_per_site_pow_2']


@memory.cache
def f_year(train_df, test_df):
    return rp(year(train_df)), rp(year(test_df)), ['hour']


@memory.cache
def f_hour(train_df, test_df):
    return rp(hour(train_df)), rp(hour(test_df)), ['hour']


@memory.cache
def f_weekday(train_df, test_df):
    return rp(weekday(train_df)), rp(weekday(test_df)), ['weekday']


def dumm(df, f):
    data = df[times]
    return csr_matrix(pd.get_dummies(f(data)))


@memory.cache
def f_hour_dummies(df):
    return dumm(df, hour)


@memory.cache
def f_weekday_dummies(df):
    return dumm(df, weekday)


@memory.cache
def f_month_dummies(df):
    return dumm(df, month)


@memory.cache
def f_year_dummies(df):
    return dumm(df, year)


@memory.cache
def f_month(train_df, test_df):
    return rp(month(train_df)), rp(month(test_df)), ['month']


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


@memory.cache
def extract_time_features(df):
    data = df[times]
    day_offset = 24
    month_offset = day_offset + 7
    morning_offset = month_offset + 12
    evening_offset = morning_offset + 1
    row_size = evening_offset + 2
    values = []

    for _, row in data.iterrows():
        time = row[times[0]]

        r = np.zeros(row_size)
        r[time.hour] += 1
        r[day_offset + time.dayofweek] += 1
        r[month_offset + time.month] += 1
        r[morning_offset] = time.hour < 11
        r[evening_offset] = time.hour > 19
        values.append(r[1:])

    return csr_matrix(values)


@memory.cache
def extract_year_month(df):
    data = df[times]
    time = times[0]
    values = [row[time].year * 100 + row[time].month for _, row in data.iterrows()]
    series = pd.Series(values)
    return csr_matrix(pd.get_dummies(series))


@memory.cache
def extract_part_of_day(df):
    data = df[times]
    time = times[0]
    values = [row[time].hour // 6 for _, row in data.iterrows()]
    series = pd.Series(values)
    return csr_matrix(pd.get_dummies(series))


@memory.cache
def extract_weekend(df):
    data = df[times]
    time = times[0]
    values = [[row[time].dayofweek > 4] for _, row in data.iterrows()]
    return csr_matrix(values)


@memory.cache
def extract_duration(df):
    data = df[times]
    values = []
    time = times[0]

    for _, row in data.iterrows():

        first = row[time]
        last = first

        for t, check in zip(times, row.values == np.datetime64('NaT')):
            if check:
                break
            else:
                last = row[t]

        values.append([np.log1p(last.minute - first.minute)])

    return csr_matrix(np.nan_to_num(values))


@memory.cache
def extract_week(df):
    data = df[times]
    time = times[0]
    values = []

    for _, row in data.iterrows():
        r = np.zeros(53)
        r[row[time].week] = 1
        values.append(r)

    return csr_matrix(values)
