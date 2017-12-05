import pandas as pd

def prepare_data_rp_entity_embedding_data():
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

def prepare_rp_rbf_data(cluster):
    train, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()

    name = ('cluster_rbf_%d') % cluster

    import pickle, os
    cluster_rbf_25_train = pickle.load(open(os.path.join('../from_scratch', name+'train.pkl'), 'rb'))
    cluster_rbf_25_test = pickle.load(open(os.path.join('../from_scratch', name+'test.pkl'), 'rb'))
    import numpy as np
    train_cat = np.hstack([train.values, cluster_rbf_25_train])
    test_cat = np.hstack([test.values, cluster_rbf_25_test])

    print (train_cat.shape, test_cat.shape)

    return train_cat, test_cat, train_id, test_id, y


train, test, train_id, test_id, y = prepare_rp_rbf_data(25)

print (train.shape, test.shape)


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
        'learning_rate': 0.07,
        'n_estimators': 1000,
        'silent': 1,
        'seed': 1024,

        'colsample_bytree': 0.97,
        'gamma': 4.6,
        'max_depth': 3,
        'min_child_weight': 19,
        'subsample':  0.93,

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
        'colsample_bytree': 1,
        'gamma': 9.9,
        'max_depth': 3,
        'min_child_weight': 1.4,
        'subsample':  0.85,
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
        'colsample_bytree': 0.99,
        'gamma': 8.5,
        'max_depth': 4,
        'min_child_weight': 19,
        'subsample':  1,
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
        'colsample_bytree': 0.98,
        'gamma': 9.9,
        'max_depth': 3,
        'min_child_weight': 12,
        'subsample':  0.836,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
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
        'colsample_bytree': 0.98,
        'gamma': 0.32,
        'max_depth': 3,
        'min_child_weight': 17,
        'subsample':  1,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
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
        'colsample_bytree': 0.78,
        'gamma': 7.35,
        'max_depth': 3,
        'min_child_weight': 19,
        'subsample':  0.7,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
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
        'colsample_bytree': 0.456,
        'gamma': 9.9,
        'max_depth': 4,
        'min_child_weight': 12,
        'subsample':  0.98,
        'n_jobs': 8,
    },
    'metric': gini,
    'bag': 4,
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

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train, y,
                            test, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=model['split'], n_folds=model['fold'], n_bags=model['bag'],
                            early_stop=early_stop, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'-rp-rbf', mean=mean, overall_mean=full_score, std=std, bag=-1)

