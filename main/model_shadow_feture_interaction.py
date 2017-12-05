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

from basic_processing import *
from base_model import *

trn_df, sub_df, train_id, test_id, y = read_data()

import os
if os.path.exists('../input/train.csv'):
    target = pd.read_csv('../input/train.csv')['target']
else:
    target = pd.read_csv('../dropcal_one_hot/train.csv')['target']


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
from basic_processing import *
from base_model import *

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
                                             # ("ps_car_13", "ps_ind_17_bin"),
                                             # ("ps_ind_05_cat", "ps_ind_17_bin")
                                            ])

f_cats = [f for f in trn_df.columns if "_cat" in f]

for f in f_cats:
    trn_df[f + "_avg"], sub_df[f + "_avg"] = target_encode(trn_series=trn_df[f],
                                                           tst_series=sub_df[f],
                                                           target=target,
                                                           min_samples_leaf=200,
                                                           smoothing=10,
                                                           noise_level=0)


train = trn_df.values
test = sub_df.values

print (train.shape, test.shape, target.shape)
# Value |   colsample_bytree |     gamma |   max_depth |   min_child_weight |   subsample |
# [35m   0.35368[0m |
# [32m            0.4719[0m |
# [32m   9.6957[0m |
# [32m     3.3838[0m |
# [32m           10.3150[0m |
# [32m     0.5351[0m |


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from rgf.sklearn import RGFClassifier
import sys

from numba import jit
@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
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

model_set = {}
model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.07,
        'n_estimators': 2000,
        'max_depth': 9,
        'scale_pos_weight': 1.0,
        'seed': 1024,

        'subsample': 0.54,
        'colsample_bytree': 0.35,
        'gamma':0.47,

        'min_child_weight': 10,
        # 'reg_lambda': 1.3,
        # 'reg_alpha': 8.0,
        # 'missing': -1,
        'n_jobs': 7,
    },
    'metric': eval_gini,
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

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging(train, y,
                            test, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=1, early_stop=50, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'-interaction', mean=mean, overall_mean=full_score, std=std, bag=-1)