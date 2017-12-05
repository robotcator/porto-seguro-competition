import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

import gc
gc.enable()

trn_df = pd.read_csv("../input/train.csv", index_col=0)
sub_df = pd.read_csv("../input/test.csv", index_col=0)

target = trn_df["target"]
y = target.values
del trn_df["target"]

train_features = [
    "ps_car_13",  # : 1571.65 / shadow  609.23
    "ps_reg_03",  # : 1408.42 / shadow  511.15
    "ps_ind_05_cat",  # : 1387.87 / shadow   84.72
    "ps_ind_03",  # : 1219.47 / shadow  230.55
    "ps_ind_15",  # :  922.18 / shadow  242.00
    "ps_reg_02",  # :  920.65 / shadow  267.50
    "ps_car_14",  # :  798.48 / shadow  549.58
    "ps_car_12",  # :  731.93 / shadow  293.62
    "ps_car_01_cat",  # :  698.07 / shadow  178.72
    "ps_car_07_cat",  # :  694.53 / shadow   36.35
    "ps_ind_17_bin",  # :  620.77 / shadow   23.15
    "ps_car_03_cat",  # :  611.73 / shadow   50.67
    "ps_reg_01",  # :  598.60 / shadow  178.57
    "ps_car_15",  # :  593.35 / shadow  226.43
    "ps_ind_01",  # :  547.32 / shadow  154.58
    "ps_ind_16_bin",  # :  475.37 / shadow   34.17
    "ps_ind_07_bin",  # :  435.28 / shadow   28.92
    "ps_car_06_cat",  # :  398.02 / shadow  212.43
    "ps_car_04_cat",  # :  376.87 / shadow   76.98
    "ps_ind_06_bin",  # :  370.97 / shadow   36.13
    "ps_car_09_cat",  # :  214.12 / shadow   81.38
    "ps_car_02_cat",  # :  203.03 / shadow   26.67
    "ps_ind_02_cat",  # :  189.47 / shadow   65.68
    "ps_car_11",  # :  173.28 / shadow   76.45
    "ps_car_05_cat",  # :  172.75 / shadow   62.92
    "ps_calc_09",  # :  169.13 / shadow  129.72
    "ps_calc_05",  # :  148.83 / shadow  120.68
    "ps_ind_08_bin",  # :  140.73 / shadow   27.63
    "ps_car_08_cat",  # :  120.87 / shadow   28.82
    "ps_ind_09_bin",  # :  113.92 / shadow   27.05
    "ps_ind_04_cat",  # :  107.27 / shadow   37.43
    "ps_ind_18_bin",  # :   77.42 / shadow   25.97
    "ps_ind_12_bin",  # :   39.67 / shadow   15.52
    "ps_ind_14",  # :   37.37 / shadow   16.65
    "ps_car_11_cat"  # Very nice spot from Tilii : https://www.kaggle.com/tilii7
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]
import time
from sklearn.preprocessing import LabelEncoder

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60))
    print('\r' * 75)
    trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
    sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
    trn_df[name1] = lbl.transform(list(trn_df[name1].values))
    sub_df[name1] = lbl.transform(list(sub_df[name1].values))

    train_features.append(name1)

trn_df = trn_df[train_features]
sub_df = sub_df[train_features]

def feature_interact(train, test, interaction_pairs, interaction_method='_', suffix=''):
    # interaction_pairs are two feature name in train's feature
    from sklearn.preprocessing import LabelEncoder
    import logging

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

trn_df, sub_df = feature_interact(trn_df, sub_df, [("ps_car_13", "ps_ind_05_cat"),
                                             ("ps_ind_05_cat", "ps_reg_03"),
                                             ("ps_car_13", "ps_ind_17_bin"),
                                             ("ps_ind_05_cat", "ps_ind_17_bin")])

f_cats = [f for f in trn_df.columns if "_cat" in f]

for f in f_cats:
    trn_df[f + "_avg"], sub_df[f + "_avg"] = target_encode(trn_series=trn_df[f],
                                                           tst_series=sub_df[f],
                                                           target=target,
                                                           min_samples_leaf=200,
                                                           smoothing=10,
                                                           noise_level=0)


X = trn_df.values
test = sub_df.values

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
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)