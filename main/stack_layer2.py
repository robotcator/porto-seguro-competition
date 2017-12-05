import pandas as pd
import os

from basic_processing import *
from base_model import *

train, test, train_id, test_id, y = read_data()

prefix = './'
prefix = './'
l2_pred = [
    ('20171118-0548-lr1-stack--0.28779-0.28733-0.00804-train.csv',
     '20171118-0548-lr1-stack--0.28779-0.28733-0.00804-test.csv'),
    ('20171118-0553-lr2-stack-0.28764-0.28723-0.00803-train.csv',
     '20171118-0553-lr2-stack-0.28764-0.28723-0.00803-test.csv'),
    ('20171118-0823-lr1-stack-0.28989-0.28942-0.00792-train.csv',
     '20171118-0823-lr1-stack-0.28989-0.28942-0.00792-test.csv'),
    ('20171118-0830-xgbl-stack-0.29014-0.28950-0.00791-train.csv',
     '20171118-0830-xgbl-stack-0.29014-0.28950-0.00791-test.csv'),

    ('20171119-0915-lr1-stack-0.29889-0.29885-0.00812-train.csv',
     '20171119-0915-lr1-stack-0.29889-0.29885-0.00812-test.csv'),
    ('20171119-0916-lr2-stack-0.29904-0.29899-0.00809-train.csv',
     '20171119-0916-lr2-stack-0.29904-0.29899-0.00809-test.csv'),
    ('20171119-0918-xgbl-stack-0.29879-0.29874-0.00823-train.csv',
     '20171119-0918-xgbl-stack-0.29879-0.29874-0.00823-test.csv'),
    # ('20171119-0803-gp-orgi-0.00000-0.29909-0.00000-train.csv',
    #  '20171119-0803-gp-orgi-0.00000-0.29909-0.00000-test.csv')
]

l3_train = pd.DataFrame()
l3_test = pd.DataFrame()

for idx, item in enumerate(l2_pred):
    print (item[0], item[1])
    train_tmp = pd.read_csv(os.path.join(prefix, item[0]))
    test_tmp = pd.read_csv(os.path.join(prefix, item[1]))
    #     print (train_tmp.columns, test_tmp.columns)

    l3_train['model' + str(idx)] = train_tmp[train_tmp.columns[1]]
    l3_test['model' + str(idx)] = test_tmp[test_tmp.columns[1]]

print (l3_train.shape, l3_test.shape, y.shape)


model_set = {}

model_set['lr1'] = {
    'name': 'lr1',
    'mdl_type': 'lr',
    'param': {
        'penalty': 'l2',
        'C': 2,
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

from base_model import *
from basic_processing import *
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from rgf.sklearn import RGFClassifier
import sys

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

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(l3_train.values, y,
                            l3_test.values, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=3, early_stop=-1, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'-stack2', mean=mean, overall_mean=full_score, std=std, bag=-1)