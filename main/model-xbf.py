from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np


class Compose(object):
    def __init__(self, transforms_params):
        self.transforms_params = transforms_params
    def __call__(self, df):
        for transform_param in self.transforms_params:
            transform, param = transform_param[0], transform_param[1]
            df = transform(df, **param)
        return df

class Processer(object):
    @staticmethod
    def drop_columns(df, col_names):
        print('Before drop columns {0}'.format(df.shape))
        df = df.drop(col_names, axis=1)
        print('After drop columns {0}'.format(df.shape))
        return df

    @staticmethod
    def dtype_transform(df):
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int8)
        return df

    @staticmethod
    def negative_one_vals(df):
        df['negative_one_vals'] = MinMaxScaler().fit_transform(df.isnull().sum(axis=1).values.reshape(-1,1))
        return df

    @staticmethod
    def ohe(df_train, df_test, cat_features, threshold=50):
        # pay attention train & test should get_dummies together
        print('Before ohe : train {0}, test {1}'.format(df_train.shape, df_test.shape))
        combine = pd.concat([df_train, df_test], axis=0)
        for column in cat_features:
            temp = pd.get_dummies(pd.Series(combine[column]), prefix=column)
            _abort_cols = []
            for c in temp.columns:
                if temp[c].sum() < threshold:
                    print('column {0} unique value {1} less than threshold {2}'.format(c, temp[c].sum(), threshold))
                    _abort_cols.append(c)
            print('Abort cat columns : {0}'.format(_abort_cols))
            _remain_cols = [ c for c in temp.columns if c not in _abort_cols ]
            # check category number
            combine = pd.concat([combine, temp[_remain_cols]], axis=1)
            combine = combine.drop([column], axis=1)
        train = combine[:df_train.shape[0]]
        test = combine[df_train.shape[0]:]
        print('After ohe : train {0}, test {1}'.format(train.shape, test.shape))
        return train, test

from basic_processing import *
train, test, train_id, test_id, y = read_data()

## Processing and Feature Engineering
transformer_one = [
    (Processer.drop_columns, dict(col_names=train.columns[train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.dtype_transform, dict()),
]
# execute transforms pipeline
print('Transform train data')
train = Compose(transformer_one)(train)
print('Transform test data')
test = Compose(transformer_one)(test)
# execute ohe
train, test = Processer.ohe(train, test, [a for a in train.columns if a.endswith('cat')])


import gc
gc.collect()

from base_model import *
from basic_processing import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rgf.sklearn import RGFClassifier
import sys

model_set = {}



model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.04,
        # 'n_estimators': 431,
        'n_estimators': 800,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.8,
        'gamma': 0.59,
        'reg_alpha': 10.4,
        'reg_lambda': 5,
        'max_depth': 5,
        'min_child_weight': 9.15,
        'subsample':  0.8,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
    'early_stop': -1,
}

model_set['xgb2'] = {
    'name': 'xgb2',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.08,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.5,
        'gamma': 9.89,
        'max_depth': 6,
        'min_child_weight': 20,
        'subsample':  0.72,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}


model_set['xgb3'] = {
    'name': 'xgb3',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.08,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.5,
        'gamma': 8.97,
        'max_depth': 5,
        'min_child_weight': 20,
        'subsample':  0.84,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb4'] = {
    'name': 'xgb4',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.497,
        'gamma': 9.89,
        'max_depth': 6,
        'min_child_weight': 19,
        'subsample': 0.7,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb5'] = {
    'name': 'xgb5',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.52,
        'gamma': 8.97,
        'max_depth': 5,
        'min_child_weight': 13,
        'subsample': 0.83,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb6'] = {
    'name': 'xgb6',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.467,
        'gamma': 7.27,
        'max_depth': 4,
        'min_child_weight': 16,
        'subsample': 0.76,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb6'] = {
    'name': 'xgb6',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.6,
        'gamma': 3.1,
        'max_depth': 4,
        'min_child_weight': 19,
        'subsample': 0.79,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb7'] = {
    'name': 'xgb7',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.53,
        'gamma': 9.99,
        'max_depth': 6,
        'min_child_weight': 13,
        'subsample': 0.545,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb8'] = {
    'name': 'xgb8',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.949,
        'gamma': 9.28,
        'max_depth': 6,
        'min_child_weight': 19,
        'subsample': 0.94,

        'n_jobs': 7,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

if __name__ == '__main__':
    print (sys.argv)

    if len(sys.argv) >= 2:
        model = model_set[sys.argv[1]]
    else:
        model = model_set['xgb1']

    metric = model['metric']
    if model['mdl_type'] == 'xgb':
        mdl = XGBClassifier(**model['param'])
    elif model['mdl_type'] == 'lgb':
        mdl = LGBMClassifier(**model['param'])
    elif model['mdl_type'] == 'rgf':
        mdl = RGFClassifier(**model['param'])

    if model['mdl_type'] == 'xgb':
        early_stop = 50
    else:
        early_stop = -1

    stratified = False

    # split will modify the fold seed, To do add new param

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train.values, y,
                            test.values, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=stratified, n_splits=model['split'], n_folds=model['fold'],
                            n_bags=model['bag'], early_stop=early_stop, verbose=True)
    if stratified:
        write_csv(train_pred, test_pred, mdl_name=model['name']+'-xbf-strati', mean=mean, overall_mean=full_score, std=std, bag=-1)
    else:
        write_csv(train_pred, test_pred, mdl_name=model['name']+'-xbf', mean=mean, overall_mean=full_score,
                  std=std, bag=-1)
    print('Single XGBoost done')