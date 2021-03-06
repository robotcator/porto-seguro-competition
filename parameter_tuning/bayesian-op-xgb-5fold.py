# http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

def train(X, test, y, metric, kfold, seed):
    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np

    skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True).split(X, y)

    train_temp_y = np.zeros(len(X))
    test_temp_y = np.zeros(len(test))

    validation = []

    for idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X[train_index, :], X[valid_index, :]
        y_train, y_valid = y[train_index], y[valid_index]

        mdl = mdl.fit(X_train, y_train)
        valid_pred = mdl.predict_proba(X_valid)[:, 1]

        score = metric(y_valid, valid_pred)
        print ("score ", score)
        validation.append(score)

        train_temp_y[valid_index] = valid_pred
        test_pred_temp = mdl.predict_proba(test)[:, 1]
        test_temp_y += test_pred_temp

    test_temp_y /= kfold
    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    return train_temp_y, test_temp_y


def five_fold(X, y, test, metric, mdl, train_id, test_id, seed=1024 ):
    import pandas as pd

    kfold = 5

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id

    train_temp_y, test_temp_y = train(X, test, y, metric=metric, kfold=kfold, seed=seed)

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y
    print ("full score: ", metric(y, train_pred['pred_y']))

    return train_pred, test_pred

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
X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()
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

    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=random_state, shuffle=True).split(X, y)
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