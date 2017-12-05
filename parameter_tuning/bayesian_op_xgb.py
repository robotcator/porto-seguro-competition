# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from bayes_opt import BayesianOptimization
import xgboost as xgb

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):

    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    
    xgbc = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=kfolds, stratified=True,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])

    val_score = xgbc['test-auc-mean'].values[-1]
    train_score = xgbc['train-auc-mean'].values[-1]

    return (2*val_score-1)

num_rounds = 1000
random_state = 1024
num_iter = 25
init_points = 15

params = {
    'objective' : 'binary:logistic',
    'booster' : 'gbtree',
    'eta': 0.1,
    'silent': 1,
    'eval_metric': 'auc',
    'verbose_eval': True,
    'seed': random_state
}

kfolds=5
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=kfolds, random_state=random_state, shuffle=True).split(train, y) 
# skf = KFold(n_splits=kfold, random_state=seed, shuffle=True).split(train, y)
xgtrain = xgb.DMatrix(train, y)

xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                            'colsample_bytree': (0.4, 1),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'min_child_weight': (1, 20),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)
