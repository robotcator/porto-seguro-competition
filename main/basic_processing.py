import logging

TRAIN_ROW = 595212
TEST_ROW = 892816

# preprocesing part
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

def feature_interact(train, test, interaction_pairs, interaction_method='_', suffix=''):
    # interaction_pairs are two feature name in train's feature
    from sklearn.preprocessing import LabelEncoder

    for idx, item in enumerate(interaction_pairs):
        logging.info("processing feature %s %s" % (item[0], item[1]))
        name = item[0] + '|' + item[1] + suffix

        if interaction_method == '_':
            train[name] = train[item[0]].apply(lambda x: str(x)) + "_" + train[item[1]].apply(lambda x: str(x))
            test[name] = test[item[0]].apply(lambda x: str(x)) + "_" + test[item[1]].apply(lambda x: str(x))

        lbl = LabelEncoder()
        lbl.fit(list(train[name].values) + list(test[name].values))
        train[name] = lbl.transform(list(train[name].values))
        test[name] = lbl.transform(list(test[name].values))

    return train, test

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

def write_csv(train, test, mean=0.0, overall_mean=0.0, std=0.0, mdl_name='', bag=-1):
    import datetime

    name = './output/' + '%s-%s-%.5f-%.5f-%.5f-' % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), mdl_name, mean, overall_mean, std)
    # assert (train.shape == (TRAIN_ROW, 2))
    # assert (test.shape == (TEST_ROW, 2))
    if bag > 0:
        name += '-bag' + str(bag) + '-'
    train.to_csv(name + 'train.csv', index=False)
    test.to_csv(name + 'test.csv', index=False)
    return

def gini(y, pred):
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g