train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

import pandas as pd
import numpy as np

from basic_processing import *
from base_model import *
from target_encoding import *


train, test, train_id, test_id, y = read_data()
train = train[train_features]
test = test[train_features]

# feature interaction
train, test = feature_interact(train, test, combs)

# my feature interaction
train, test = feature_interact(train, test, [("ps_car_13", "ps_ind_05_cat"),
                                             ("ps_ind_05_cat", "ps_reg_03"),
                                             ("ps_car_13", "ps_ind_17_bin"),
                                             ("ps_ind_05_cat", "ps_ind_17_bin")])

import os
if os.path.exists('../input/train.csv'):
    target = pd.read_csv('../input/train.csv')['target']
else:
    target = pd.read_csv('../dropcal_one_hot/train.csv')['target']

print (train.shape, test.shape)


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from rgf.sklearn import RGFClassifier
import sys
# sys.path.append('/Users/robotcator/clone/rgf_python/include/rgf/bin')

model_set = {}
model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 400,
        'max_depth': 4,
        'scale_pos_weight': 1.6,
        'seed': 1024,

        'subsample': 0.9,
        'colsample_bytree': 0.8,

        'min_child_weight': 6,
        'reg_lambda': 1.3,
        'reg_alpha': 8.0,
        # 'missing': -1,
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
        'learning_rate': 0.07,
        'n_estimators': 400,
        'max_depth': 4,
        'scale_pos_weight': 1.8,
        'seed': 1024,

        'subsample': 0.9,
        'colsample_bytree': 0.8,

        'min_child_weight': 6,
        'reg_lambda': 1.3,
        'reg_alpha': 8.0,
        # 'missing': -1,
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
        'learning_rate': 0.07,
        'n_estimators': 400,
        'max_depth': 4,
        'scale_pos_weight': 1.8,
        'seed': 1024,

        'subsample': 0.9,
        'colsample_bytree': 0.8,

        'min_child_weight': 6,
        'reg_lambda': 1.3,
        'reg_alpha': 8.0,
        'gamma': 6,
        # 'missing': -1,
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
        'learning_rate': 0.07,
        'n_estimators': 400,
        'max_depth': 4,
        'scale_pos_weight': 1.8,
        'seed': 1024,

        'subsample': 0.8,
        'colsample_bytree': 0.8,

        'min_child_weight': 6,
        'reg_lambda': 1.3,
        'reg_alpha': 8.0,
        'gamma': 8,
        # 'missing': -1,
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
        'learning_rate': 0.04,
        'n_estimators': 430,
        'max_depth': 5,
        'scale_pos_weight': 1.8,
        'seed': 1024,

        'subsample': 0.9,
        'colsample_bytree': 0.8,

        'min_child_weight': 9.15,
        'reg_lambda': 5,
        'reg_alpha': 10.4,
        'gamma': 0.59,
        # 'missing': -1,
        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['lgb1'] = {
    'name': 'lgb1',
    'mdl_type': 'lgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.02,
        'n_estimators': 650,
        'max_bin': 10,
        'seed': 1024,
        'subsample': 0.8,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_samples': 500,
        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['lgb2'] = {
    'name': 'lgb2',
    'mdl_type': 'lgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.02,
        'n_estimators': 1000,
        'num_leaves': 16,
        'seed': 1024,
        'subsample': 0.8,
        'subsample_freq': 4,
        'colsample_bytree': 0.3,
        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['lgb3'] = {
    'name': 'lgb3',
    'mdl_type': 'lgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.02,
        'n_estimators': 800,
        'num_leaves': 18,
        'seed': 1024,
        'subsample': 0.8,
        'subsample_freq': 4,
        'colsample_bytree': 0.5,
        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['rgf1'] = {
    'name': 'rgf1',
    'mdl_type': 'rgf',
    'param': {
        'max_leaf': 1000,
        'algorithm': 'RGF',
        'loss': 'Log',
        'l2': 0.01,
        'sl2': 0.01,
        'normalize': False,
        'min_samples_leaf': 10,
        'n_iter': None,
        'opt_interval': 100,
        'learning_rate': 0.5,
        'calc_prob': "sigmoid",
        'n_jobs': -1,
        'memory_policy': "generous",
        'verbose': 1,
    },
    'metric': gini,
}

model_set['rgf2'] = {
    'name': 'rgf2',
    'mdl_type': 'rgf',
    'param': {
        'max_leaf': 1000,
        'algorithm': 'RGF_Sib',
        'loss': 'Log',
        'l2': 0.01,
        'sl2': 0.01,
        'normalize': False,
        'min_samples_leaf': 10,
        'n_iter': None,
        'opt_interval': 100,
        'learning_rate': 0.5,
        'calc_prob': "sigmoid",
        'n_jobs': -1,
        'memory_policy': "generous",
        'verbose': 1,
    },
    'metric': gini,
}

model_set['rgf3'] = {
    'name': 'rgf3',
    'mdl_type': 'rgf',
    'param': {
        'max_leaf': 1000,
        'algorithm': 'RGF',
        'loss': 'Log',
        'l2': 0.01,
        'sl2': 0.01,
        'normalize': False,
        'min_samples_leaf': 15,
        'n_iter': None,
        'opt_interval': 100,
        'learning_rate': 0.4,
        'calc_prob': "sigmoid",
        'n_jobs': -1,
        'memory_policy': "generous",
        'verbose': 1,
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

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging_with_target_encode(train, target,
                            test, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=1, early_stop=-1, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name'], mean=mean, overall_mean=full_score, std=std, bag=-1)