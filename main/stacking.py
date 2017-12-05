import pandas as pd
import os

from basic_processing import *
from base_model import *

train, test, train_id, test_id, y = read_data()


prefix = './'
l1_pred = [
    ('20171118-0252-xgb1-0.28536-0.28518-0.00849-train.csv',
     '20171118-0252-xgb1-0.28536-0.28518-0.00849-test.csv'),
    ('20171118-0301-xgb2-0.28510-0.28491-0.00812-train.csv',
     '20171118-0301-xgb2-0.28510-0.28491-0.00812-test.csv'),
    ('20171118-0314-xgb3-0.28586-0.28569-0.00754-train.csv',
     '20171118-0314-xgb3-0.28586-0.28569-0.00754-test.csv'),
    ('20171118-0400-xgb4-0.28619-0.28602-0.00828-train.csv',
     '20171118-0400-xgb4-0.28619-0.28602-0.00828-test.csv'),
    ('20171118-0411-xgb5-0.28620-0.28602-0.00784-train.csv',
     '20171118-0411-xgb5-0.28620-0.28602-0.00784-test.csv'),
    # all the same above, so
    ('20171118-0418-lgb1-0.28703-0.28682-0.00698-train.csv',
     '20171118-0418-lgb1-0.28703-0.28682-0.00698-test.csv'),
    ('20171118-0427-lgb2-0.28705-0.28680-0.00831-train.csv',
     '20171118-0427-lgb2-0.28705-0.28680-0.00831-test.csv'),
    ('20171118-0436-lgb3-0.28573-0.28550-0.00845-train.csv',
     '20171118-0436-lgb3-0.28573-0.28550-0.00845-test.csv'),

    ('20171118-0609-lgb1ohe-0.28612-0.28579-0.00842-train.csv',
     '20171118-0609-lgb1ohe-0.28612-0.28579-0.00842-test.csv'),

    ('20171118-0644-rgf1-0.28247-0.28231-0.00791-train.csv',
     '20171118-0644-rgf1-0.28247-0.28231-0.00791-test.csv'),
    # ('20171118-0654-rgf2-0.27969-0.27957-0.00655-train.csv',
    #  '20171118-0654-rgf2-0.27969-0.27957-0.00655-test.csv'),
    ('20171118-0805-rgf3-0.28373-0.28354-0.00787-train.csv',
     '20171118-0805-rgf3-0.28373-0.28354-0.00787-test.csv'),

    ('20171118-0725-xgb1-rp-0.28459-0.28427-0.00897-train.csv',
     '20171118-0725-xgb1-rp-0.28459-0.28427-0.00897-test.csv'),
    ('20171118-0734-xgb2-rp-0.28635-0.28614-0.00798-train.csv',
     '20171118-0734-xgb2-rp-0.28635-0.28614-0.00798-test.csv'),
    ('20171118-0747-xgb3-rp-0.28603-0.28566-0.00799-train.csv',
     '20171118-0747-xgb3-rp-0.28603-0.28566-0.00799-test.csv'),
    ('20171118-0755-xgb4-rp-0.28685-0.28663-0.00792-train.csv',
     '20171118-0755-xgb4-rp-0.28685-0.28663-0.00792-test.csv'),

    ('20171118-0808-lgb1-rp-0.28781-0.28754-0.00836-train.csv',
     '20171118-0808-lgb1-rp-0.28781-0.28754-0.00836-test.csv'),
    ('20171118-0812-lgb2-rp-0.28568-0.28547-0.00661-train.csv',
     '20171118-0812-lgb2-rp-0.28568-0.28547-0.00661-test.csv'),

    ('20171119-1138-lr1-gp-0.30105-0.30087-0.01017-train.csv',
     '20171119-1138-lr1-gp-0.30105-0.30087-0.01017-test.csv'),
    ('20171119-1207-lr2-gp-0.30006-0.29989-0.00968-train.csv',
     '20171119-1207-lr2-gp-0.30006-0.29989-0.00968-test.csv'),
    ('20171119-1214-lr3-gp-0.30105-0.30087-0.01017-train.csv',
     '20171119-1207-lr2-gp-0.30006-0.29989-0.00968-test.csv'),
    ('20171119-1243-lr4-gp-0.29832-0.29815-0.00976-train.csv',
     '20171119-1243-lr4-gp-0.29832-0.29815-0.00976-test.csv'),

    ('20171120-1613-xgb1-xbf-0.28519-0.28478-0.00828-train.csv',
     '20171120-1613-xgb1-xbf-0.28519-0.28478-0.00828-test.csv'),
    ('20171122-0729-xgb3-0.28400-0.28375-0.00793-train.csv',
     '20171122-0729-xgb3-0.28400-0.28375-0.00793-test.csv'),
    ('20171122-0752-xgb3-0.28313-0.28279-0.00827-train.csv',
     '20171122-0752-xgb3-0.28313-0.28279-0.00827-test.csv'),

    ('20171120-1710-xgbpascal-0.28293-0.28256-0.00780-train.csv',
     '20171120-1710-xgbpascal-0.28293-0.28256-0.00780-test.csv'),

    ('../dropcal_one_hot/xgb_train_34f_onehot_depth4_booster1000_row08_col09_score286_0.004_lambda15_gamma0_mincw_10.csv',
    '../dropcal_one_hot/xgb_test_34f_onehot_depth4_booster1000_row08_col09_score286_0.004_lambda15_gamm0_mincw_10.csv'),
    ('../dropcal_one_hot/lgb1_train_lr02_ntree600_maxbin10_row8_rowfreq10_col8_mcsample500_286_006.csv',
     '../dropcal_one_hot/lgb1_test_lr02_ntree600_maxbin10_row8_rowfreq10_col8_mcsample500_286_006.csv'),

]

l2_train = pd.DataFrame()
l2_test = pd.DataFrame()

for idx, item in enumerate(l1_pred):
    # print (item[0], item[1])
    train_tmp = pd.read_csv(os.path.join(prefix, item[0]))
    test_tmp = pd.read_csv(os.path.join(prefix, item[1]))
    #     print (train_tmp.columns, test_tmp.columns)

    l2_train['model' + str(idx)] = train_tmp[train_tmp.columns[1]]
    l2_test['model' + str(idx)] = test_tmp[test_tmp.columns[1]]

print (l2_train.shape, l2_test.shape, y.shape)


from base_model import *
from basic_processing import *
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from rgf.sklearn import RGFClassifier
import sys


model_set = {}

model_set['lr1'] = {
    'name': 'lr1',
    'mdl_type': 'lr',
    'param': {
        'penalty': 'l2',
    },
    'metric': gini,
    'bag': 2,
    'split': 1,
    'fold': 5,
}

model_set['lr11'] = {
    'name': 'lr11',
    'mdl_type': 'lr',
    'param': {
        'penalty': 'l2',
        'C': 0.8,
    },
    'metric': gini,
    'bag': 2,
    'split': 1,
    'fold': 5,
}

model_set['lr2'] = {
    'name': 'lr2',
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

model_set['xgbl'] = {
    'name': 'xgbl',
    'mdl_type': 'xgb',
    'param' : {
        'objective':'binary:logistic',
        'booster':'gblinear',
        'alpha': 0.0001,
        'lambda': 2,
        'silent':1,
        'seed': 1024
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['mlp'] = {
    'name': 'mlp',
    'mdl_type': 'mlp',
    'param': {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'early_stopping': True,
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
    elif model['mdl_type'] == 'mlp':
        mdl = MLPClassifier(**model['param'])

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(l2_train.values, y,
                            l2_test.values, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=1, early_stop=-1, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'-stack', mean=mean, overall_mean=full_score, std=std, bag=-1)