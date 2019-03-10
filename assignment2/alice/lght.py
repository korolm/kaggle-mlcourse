import lightgbm as lgb

params = {'boosting_type': 'gbdt',
          'max_depth': -1,
          'objective': 'binary',
          'nthread': 3,  # Updated from nthread
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class': 1,
          'metric': 'binary_error'}

# Create parameters to search
gridParams = {
    'learning_rate': (0.02, 0.1),
    'n_estimators': (300, 2000),
    'num_leaves': (16, 40),
    'colsample_bytree': (0.6, 0.8),
    'subsample': (0.8, 1)
}

def get_est_params():
    return gridParams

def create_est(kwargs):
    return lgb.LGBMClassifier(learning_rate=kwargs.get('learning_rate'),
                             n_estimators=int(kwargs.get('n_estimators')),
                             num_leaves=int(kwargs.get('num_leaves')),
                             colsample_bytree=kwargs.get('colsample_bytree'),
                             subsample=kwargs.get('subsample'),
                             boosting_type='gbdt',
                             objective='binary',
                             n_jobs=3,  # Updated from 'nthread'
                             random_state=17,
                             silent=True,
                             max_depth=params['max_depth'],
                             max_bin=params['max_bin'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])
