import logging
import numpy as np
import pandas as pd
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


train, test, train_id, test_id, y = read_data()


def recon(reg):
    integer = int(np.round((40 * reg) ** 2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A) // 31
    return A, M


train['ps_reg_A'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
train['ps_reg_A'].replace(19, -1, inplace=True)
train['ps_reg_M'].replace(51, -1, inplace=True)

test['ps_reg_A'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
test['ps_reg_M'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])
test['ps_reg_A'].replace(19, -1, inplace=True)
test['ps_reg_M'].replace(51, -1, inplace=True)

d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
d_skew = train.skew(axis=0)

one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}


def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)

    for c in dcol:
        if '_bin' not in c:  # standard arithmetic
            df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


from multiprocessing import Pool, cpu_count
def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close();
    p.join()
    print('After Shape: ', df.shape)
    return df


train = multi_transform(train)
test = multi_transform(test)
print (train.shape, test.shape)

import gc

gc.collect()

# prepare data
# X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()

# X, test, train_id, test_id, y = prepare_ohe_data()

X = train.values
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
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'silent': 1,
    'seed': random_state
}
metric = gini

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params['max_depth'] = int(max_depth)
    params['gamma'] = max(gamma, 0)
    params['min_child_weight'] = int(min_child_weight)

    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)

    mdl = XGBClassifier(**params)

    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np

    kfold = 5
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
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)