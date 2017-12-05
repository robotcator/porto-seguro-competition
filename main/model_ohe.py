import numpy as np
import pandas as pd

from basic_processing import *
from base_model import *

train, test, train_id, test_id, y = read_data()
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = drop_columns(train, col_to_drop)
test = drop_columns(test, col_to_drop)

cat_feature = [item for item in train.columns if item.endswith("cat")]
train, test = ohe(train, test, cat_features=cat_feature)

print (train.shape, test.shape)
print (train.columns)


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier
import sys

model_set = {}


model_set['lgb1'] = {
    'name': 'lgb1',
    'mdl_type': 'lgb',
    'param': {
        'objective' : 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.02,
        'n_estimators': 600,
        'max_bin': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'subsample_freq': 10,
        'min_child_samples': 500,
        'random_state': 1024,
    },
    'metric': gini,
}

model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.93,
        'gamma': 8.8,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb2'] = {
    'name': 'xgb2',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.4,
        'gamma': 9.8,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.9,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb3'] = {
    'name': 'xgb3',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.45,
        'gamma': 6.5,
        'max_depth': 6,
        'min_child_weight': 19,
        'subsample': 0.98,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb4'] = {
    'name': 'xgb4',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.4,
        'gamma': 5.7,
        'max_depth': 6,
        'min_child_weight': 15,
        'subsample': 1,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb5'] = {
    'name': 'xgb5',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.93,
        'gamma': 9.4,
        'max_depth': 6,
        'min_child_weight': 19,
        'subsample': 1,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb6'] = {
    'name': 'xgb6',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.455,
        'gamma': 10,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 1,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb7'] = {
    'name': 'xgb7',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.41,
        'gamma': 9.4,
        'max_depth': 6,
        'min_child_weight': 10.8,
        'subsample': 0.98,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb8'] = {
    'name': 'xgb8',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.418,
        'gamma': 9.7,
        'max_depth': 6,
        'min_child_weight': 17,
        'subsample': 0.978,

        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['xgb9'] = {
    'name': 'xgb9',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.54,
        'gamma': 3.4,
        'max_depth': 4,
        'min_child_weight': 16,
        'subsample': 1,

        'n_jobs': 7,
    },
    'metric': gini,
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
    elif model['mdl_type'] == 'gbc':
        mdl = GradientBoostingClassifier(**model['param'])

    if model['mdl_type'] == 'xgb':
        early_stop = 50
    else:
        early_stop = 1

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train.values, y,
                            test.values, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=1, early_stop=early_stop, verbose=False, inc=0)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'ohe', mean=mean, overall_mean=full_score, std=std, bag=-1)