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

import logging
TRAIN_ROW = 595212
TEST_ROW = 892816

def dtype_transform(df):
    import numpy as np
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int8)
    return df

def drop_columns(df, col_names):
    logging.info('Before drop columns {0}'.format(df.shape))
    df = df.drop(col_names, axis=1)
    logging.info('After drop columns {0}'.format(df.shape))
    return df

def ohe(df_train, df_test, cat_features, threshold=50):
    # https: // www.kaggle.com / xbf6xbf / single - xgb - lb284 / code
    import pandas as pd
    # pay attention train & test should get_dummies together
    logging.info('Before ohe : train {0}, test {1}'.format(df_train.shape, df_test.shape))
    combine = pd.concat([df_train, df_test], axis=0)
    for column in cat_features:
        temp = pd.get_dummies(pd.Series(combine[column]), prefix=column)
        _abort_cols = []
        # why using this
        # for c in temp.columns:
        #     if temp[c].sum() < threshold:
        #         print('column {0} unique value {1} less than threshold {2}'.format(c, temp[c].sum(), threshold))
        #         _abort_cols.append(c)
        # print('Abort cat columns : {0}'.format(_abort_cols))
        _remain_cols = [c for c in temp.columns if c not in _abort_cols]
        # check category number
        combine = pd.concat([combine, temp[_remain_cols]], axis=1)
        combine = combine.drop([column], axis=1)

    train = combine[:df_train.shape[0]]
    test = combine[df_train.shape[0]:]
    logging.info('After ohe : train {0}, test {1}'.format(train.shape, test.shape))
    return train, test

def read_data():
    logging.info("Reading data")
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

    assert (len(train_id) == TRAIN_ROW)
    assert (len(test_id) == TEST_ROW)

    train = train.drop(['id', 'target'], axis=1)
    test = test.drop(['id'], axis=1)

    train = dtype_transform(train)
    test = dtype_transform(test)

    logging.info("Finish reading data")
    return train, test, train_id, test_id, y


def prepare_ohe_data():
    train, test, train_id, test_id, y = read_data()
    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = drop_columns(train, col_to_drop)
    test = drop_columns(test, col_to_drop)

    cat_feature = [item for item in train.columns if item.endswith("cat")]
    train, test = ohe(train, test, cat_features=cat_feature)
    return train, test, train_id, test_id, y

# prepare data
# X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()

X, test, train_id, test_id, y = prepare_ohe_data()
X = X.values
test = test.values

# define the optimization
from bayes_opt import BayesianOptimization
import xgboost as xgb
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
    'objective' : 'binary:logistic',
    # 'booster' : 'gbtree',
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'silent': 1,
    'n_jobs': 8,
    'seed': random_state
}
metric = gini

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma, scale_pos_weight=1):
    params['max_depth'] = int(max_depth)
    params['gamma'] = max(gamma, 0)
    params['min_child_weight'] = int(min_child_weight)

    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['scale_pos_weight'] = scale_pos_weight

    mdl = XGBClassifier(**params)

    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np

    kfold = 5
    # skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True).split(X, y)
    skf = KFold(n_splits=kfold, random_state=random_state, shuffle=True).split(X, y)
    train_temp_y = np.zeros(len(X))

    validation = []

    for idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X[train_index, :], X[valid_index, :]
        y_train, y_valid = y[train_index], y[valid_index]

        mdl = mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=50, verbose=False)
        valid_pred = mdl.predict_proba(X_valid)[:, 1]

        score = metric(y_valid, valid_pred)
        print ("score :", score)
        validation.append(score)

        train_temp_y[valid_index] = valid_pred

    full_score = metric(y, train_temp_y)
    print ("mean/ std/ full score", np.mean(validation), np.std(validation), full_score)
    return np.mean(validation)

xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                            'colsample_bytree': (0.4, 1),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'min_child_weight': (1, 20),
                                            'scale_pos_weight': (1, 2),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)