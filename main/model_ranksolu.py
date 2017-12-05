
from base_model import *
from basic_processing import *
from target_encoding import *
from sklearn.model_selection import KFold, StratifiedKFold
from numba import jit

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    preds -= preds.min()
    preds / preds.max()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def read_data():
    import pandas as pd
    import os
    if os.path.exists('../input/train.csv'):
        train = pd.read_csv('../input/train.csv')
        test = pd.read_csv('../input/test.csv')
    else:
        # local
        train = pd.read_csv('../dropcal_one_hot/train.csv')
        test = pd.read_csv('../dropcal_one_hot/test.csv')

    # Preprocessing
    y = train['target'].values
    train_id = train['id'].values
    test_id = test['id'].values

    return train, test, train_id, test_id, y

train, test, y, train_id, test_id = read_data()

test.insert(1,'target',0)
print (train.shape, test.shape)

x = pd.concat([train,test])
x = x.reset_index(drop=True)
unwanted = x.columns[x.columns.str.startswith('ps_calc_')]
x.drop(unwanted,inplace=True,axis=1)

x.loc[:,'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50,labels=False)
x.loc[:,'ps_car_12'] = pd.cut(x['ps_car_12'], 50,labels=False)
x.loc[:,'ps_car_13'] = pd.cut(x['ps_car_13'], 50,labels=False)
x.loc[:,'ps_car_14'] =  pd.cut(x['ps_car_14'], 50,labels=False)
x.loc[:,'ps_car_15'] =  pd.cut(x['ps_car_15'], 50,labels=False)

test = x[train.shape[0]:].copy()
train = x[:train.shape[0]].copy()


def five_fold(train, test, mdl):
    features = train.columns[2:]
    ranktestpreds = np.zeros(test.shape[0])
    trainpreds = np.zeros(train.shape[0])
    validation = []

    kf = KFold(n_splits=5, shuffle=True, random_state=1024)
    for i, (train_index, test_index) in enumerate(kf.split(list(train.index))):
        print('Fold: ', i)
        myfeatures = list(features[:])
        blindtrain = train.iloc[test_index].copy()
        vistrain = train.iloc[train_index].copy()

        mytest = test.copy()
        for column in features:
            vis, blind, tst = target_encode(trn_series=vistrain[column],
                                            val_series=blindtrain[column],
                                            tst_series=mytest[column],
                                            target=vistrain.target,
                                            min_samples_leaf=200,
                                            smoothing=10,
                                            noise_level=0)
            vistrain['te_' + column] = vis
            blindtrain['te_' + column] = blind
            mytest['te_' + column] = tst
            myfeatures = myfeatures + list(['te_' + column])

        eval_set = [(blindtrain[myfeatures], blindtrain.target)]
        model = mdl.fit(vistrain[myfeatures], vistrain.target,
                        eval_set=eval_set,
                        eval_metric=gini_xgb,
                        early_stopping_rounds=50,
                        verbose=False)

        print("  Best N trees = ", model.best_ntree_limit)
        print("  Best gini = ", model.best_score)
        trainpreds[test_index] = model.predict_proba(blindtrain[myfeatures], ntree_limit=model.best_ntree_limit)[:, 1]
        score = eval_gini(blindtrain.target, trainpreds[test_index])
        print("  Best gini = ", score)
        ranktestpreds += model.predict_proba(mytest[myfeatures], ntree_limit=model.best_ntree_limit)[:, 1]
        validation.append(score)

    ranktestpreds /= 5
    ranktestpreds -= ranktestpreds.min()
    ranktestpreds /= ranktestpreds.max()

    trainpreds -= trainpreds.min()
    trainpreds /= trainpreds.max()

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id

    train_pred['pred_y'] = ranktestpreds
    test_pred['target'] = trainpreds

    mean = np.mean(validation)
    std = np.std(validation)
    full_score = eval_gini(train.target, trainpreds)

    return train_pred, test_pred, mean, std, full_score


model_set = {}
model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'rank:pairwise',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'colsample_bytree': 0.4488,
        'gamma': 2.511,
        'max_depth': 3,
        'min_child_weight': 19,
        'scale_pos_weight': 1.99,
        'subsample': 0.9831,

        'silent': 1,
        'n_jobs': 8,
        'seed': 99,
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb2'] = {
    'name': 'xgb2',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'rank:pairwise',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'colsample_bytree': 0.554,
        'gamma': 9.6,
        'max_depth': 3,
        'min_child_weight': 19,
        'scale_pos_weight': 1.5,
        'subsample': 0.92,

        'silent': 1,
        'n_jobs': 8,
        'seed': 99,
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

model_set['xgb3'] = {
    'name': 'xgb3',
    'mdl_type': 'xgb',
    'param': {
        'objective' : 'rank:pairwise',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'colsample_bytree': 0.407,
        'gamma': 2.17,
        'max_depth': 4,
        'min_child_weight': 9.4,
        'scale_pos_weight': 1.3,
        'subsample': 0.76,

        'silent': 1,
        'n_jobs': 8,
        'seed': 99,
    },
    'metric': gini,
    'bag': 3,
    'split': 1,
    'fold': 5,
}

if __name__ == '__main__':
    import sys
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

    if model['mdl_type'] == 'xgb':
        early_stop = 50
    else:
        early_stop = 1

    train_pred, test_pred, mean, std, full_score = five_fold(train, test, mdl)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'rankpair', mean=mean, overall_mean=full_score, std=std, bag=-1)



