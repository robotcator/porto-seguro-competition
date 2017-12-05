import pandas as pd

def prepare_data_rp_entity_embeding_data():
    import os
    if os.path.exists('../input/train.csv'):
        train = pd.read_csv('../input/train.csv')
        test = pd.read_csv('../input/test.csv')
    else:
        train = pd.read_csv('../dropcal_one_hot/train.csv')
        test = pd.read_csv('../dropcal_one_hot/test.csv')

    y = train['target'].values
    train_id = train['id'].values
    test_id = test['id'].values

    train = pd.read_csv('../input/train_random_project_feature.csv')
    test = pd.read_csv('../input/test_random_project_feature.csv')

    print (train.shape, test.shape)
    import gc
    gc.collect()
    return train, test, train_id, test_id, y

train, test, train_id, test_id, y = prepare_data_rp_entity_embeding_data()

print (train.shape, test.shape)


from base_model import *
from basic_processing import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
    'split': 1,
    'fold': 5,
}

model_set['lgb2'] = {
    'name': 'lgb2',
    'mdl_type': 'lgb',
    'param': {
        'objective' : 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.08,
        'n_estimators': 600,

        'colsample_bytree': 0.64,
        'max_bin': 179,
        'max_depth': 3,
        'min_child_weight': 16,
        'subsample': 0.76,
        'subsample_freq': 7,

        'random_state': 1024,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
    'split': 1,
    'fold': 5,
}

model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.4624,
        'gamma': 9.83,
        'max_depth': 5,
        'min_child_weight': 18,
        'subsample':  1,

        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
    'split': 1,
    'fold': 5,
}

model_set['xgb2'] = {
    'name': 'xgb2',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.94,
        'gamma': 6.1,
        'max_depth': 4,
        'min_child_weight': 10,
        'subsample':  0.87,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
    'split': 1,
    'fold': 5,
}

model_set['xgb3'] = {
    'name': 'xgb3',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.73,
        'gamma': 4.9,
        'max_depth': 3,
        'min_child_weight': 15,
        'subsample':  0.79,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
    'split': 1,
    'fold': 5,
}

model_set['xgb4'] = {
    'name': 'xgb4',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.53,
        'gamma': 0.056,
        'max_depth': 3,
        'min_child_weight': 17,
        'subsample':  0.948,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb5'] = {
    'name': 'xgb5',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.75,
        'gamma': 5.4,
        'max_depth': 3,
        'min_child_weight': 9,
        'subsample':  0.9,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb6'] = {
    'name': 'xgb6',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.966,
        'gamma': 9.9,
        'max_depth': 5,
        'min_child_weight': 14,
        'subsample':  1,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb7'] = {
    'name': 'xgb7',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.98,
        'gamma': 0.16,
        'max_depth': 3,
        'min_child_weight': 19,
        'subsample':  0.73,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb8'] = {
    'name': 'xgb8',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.95,
        'gamma': 9.9,
        'max_depth': 6,
        'min_child_weight': 9,
        'subsample':  1,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb9'] = {
    'name': 'xgb9',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.4268,
        'gamma': 8.2,
        'max_depth': 4,
        'min_child_weight': 13,
        'subsample':  0.993,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb10'] = {
    'name': 'xgb10',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'binary:logistic',
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,
        'colsample_bytree': 0.95,
        'gamma': 9.9,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample':  0.87,
        'n_jobs': 8
    },
    'metric': gini,
    'bag': 3,
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

    # split will modify the fold seed, To do add new param

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train.values, y,
                            test.values, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=model['split'], n_folds=model['fold'], n_bags=model['bag'],
                            early_stop=early_stop, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'-rp', mean=mean, overall_mean=full_score, std=std, bag=-1)

