# http://lightgbm.readthedocs.io/en/latest/Python-API.html

import logging

def prepare_data_rp_entity_embedding_data():
    import pandas as pd
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # Preprocessing
    target = train['target']

    y = train['target'].values
    train_id = train['id'].values
    test_id = test['id'].values

    train = pd.read_csv('../input/train_random_project_feature.csv')
    test = pd.read_csv('../input/test_random_project_feature.csv')

    print (train.shape, test.shape)
    import gc
    gc.collect()
    return train, test, train_id, test_id, y

# prepare data
logging.info("begin prepare data")
X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()
X = X.values
logging.info("end prepare data")

# define the optimization
from bayes_opt import BayesianOptimization
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

num_rounds = 1000
random_state = 1024
num_iter = 25
init_points = 15

from sklearn.metrics import auc, roc_auc_score, roc_curve
def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g

params = {
    'objective' : 'binary',
    'learning_rate': 0.08,
    'n_estimators': 1000,
    'silent': 1,
    'seed': random_state
}

metric = gini

def xgb_evaluate(min_child_weight,
                 colsample_bytree, num_leaves,
                 max_depth, max_bin,
                 subsample, subsample_freq):

    params['max_depth'] = int(max_depth)
    if num_leaves <= 2**params['max_depth']:
        params['num_leaves'] = int(num_leaves)
    else:
        params['num_leaves'] = 2**params['max_depth']

    params['max_bin'] = int(max_bin)
    params['min_child_weight'] = int(max(min_child_weight, 0))
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['subsample_freq'] = int(max(min_child_weight, 0))

    mdl = LGBMClassifier(**params)

    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True).split(X, y)
    train_temp_y = np.zeros(len(X))

    validation = []

    for idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X[train_index, :], X[valid_index, :]
        y_train, y_valid = y[train_index], y[valid_index]

        mdl = mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc", early_stopping_rounds=50)
        valid_pred = mdl.predict_proba(X_valid)[:, 1]

        score = metric(y_valid, valid_pred)
        validation.append(score)
        print ("fold %d: score %f" % (idx, score))

        train_temp_y[valid_index] = valid_pred

    full_score = metric(y, train_temp_y)
    print ("mean/ std/ full score", np.mean(validation), np.std(validation), full_score)
    return np.mean(validation)

xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                            'max_bin': (1, 1000),
                                            'num_leaves': (8, 128),
                                            'min_child_weight': (1, 20),
                                            'colsample_bytree': (0.4, 1),
                                            'subsample': (0.5, 1),
                                            'subsample_freq': (1, 10),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)

