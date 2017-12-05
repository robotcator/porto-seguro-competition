# https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284/code

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from basic_processing import *
from base_model import *

train, test, train_id, test_id, y = read_data()

def recon(reg):
    integer = int(np.round((40*reg)**2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A) // 31
    return A, M

train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19, -1, inplace=True)
train['ps_reg_M'].replace(51, -1, inplace=True)

test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19,-1, inplace=True)
test['ps_reg_M'].replace(51,-1, inplace=True)


d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)

one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)

    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

train = multi_transform(train)
test = multi_transform(test)
print (train.shape, test.shape)

import gc
gc.collect()

from base_model import *
from basic_processing import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rgf.sklearn import RGFClassifier
from sklearn.linear_model import LogisticRegression
import sys

model_set = {}
model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.025,
        'n_estimators': 5000,
        'max_depth': 4,
        'silent': 1,
        'scale_pos_weight': 1,
        'seed': 99,

        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'min_child_weight': 100,
        'n_jobs': 8,
    },
    'metric': gini,
}

model_set['xgb2'] = {
    'name': 'xgb2',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.52,
        'gamma': 0.212,
        'max_depth': 6,
        'min_child_weight': 6,
        'subsample':  0.85,
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
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.909,
        'gamma': 3.8,
        'max_depth': 5,
        'min_child_weight': 14,
        'subsample':  0.83,
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
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.51,
        'gamma': 1.5,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample':  0.85,
        'n_jobs': 8,
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
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.62,
        'gamma': 7.6,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample':  0.63,
        'n_jobs': 8,
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
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.683,
        'gamma': 2.8,
        'max_depth': 6,
        'min_child_weight': 6,
        'subsample':  0.86,
        'n_jobs': 8,
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
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.824,
        'gamma': 5.4,
        'max_depth': 6,
        'min_child_weight': 10,
        'subsample':  0.81,
        'n_jobs': 8,
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
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 1,
        'gamma': 9.85,
        'max_depth': 6,
        'min_child_weight': 20,
        'subsample':  0.5,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 1,
    'split': 1,
    'fold': 5,
}

model_set['xgb9'] = {
    'name': 'xgb9',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.934,
        'gamma': 5.8,
        'max_depth': 6,
        'min_child_weight': 17,
        'subsample':  0.5,
        'n_jobs': 8,
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
    elif model['mdl_type'] == 'lr':
        mdl = LogisticRegression(**model['param'])
    elif model['mdl_type'] == 'xgbl':
        mdl = XGBRegressor(**model['param'])

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train.values, y, test.values, train_id, test_id,
                            metric, mdl, mdl_type=model['mdl_type'], seed=1024, stratified=False,
                                            n_splits=1, n_folds=5, n_bags=1, early_stop=70, inc=0)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'pascal', mean=mean, overall_mean=full_score, std=std)
