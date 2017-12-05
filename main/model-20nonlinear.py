import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc
import subprocess
import glob

import time

start_time = time.time()
tcurrent = start_time

np.random.seed(31143)


# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,  # Revised to encode validation series
                  val_series=None,
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
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series,
                                                                                                   noise_level)


# Read data
train_df = pd.read_csv('../input/train.csv')  # .iloc[0:200,:]
test_df = pd.read_csv('../input/test.csv')

# ---- begin FEATURE ENGINEERING: NONLINEAR feature engineering by Leandro dos Santos Coelho
# train
train_df['v001'] = train_df["ps_ind_03"] + train_df["ps_ind_14"] + np.square(train_df["ps_ind_15"])
train_df['v002'] = train_df["ps_ind_03"] + train_df["ps_ind_14"] + np.tanh(train_df["ps_ind_15"])
train_df['v003'] = train_df["ps_reg_01"] + train_df["ps_reg_02"] ** 3 + train_df["ps_reg_03"]
train_df['v004'] = train_df["ps_reg_01"] ** 2.15 + np.tanh(train_df["ps_reg_02"]) + train_df["ps_reg_03"] ** 3.1
train_df['v005'] = train_df["ps_calc_01"] + train_df["ps_calc_13"] + np.tanh(train_df["ps_calc_14"])
train_df['v006'] = train_df["ps_car_13"] + np.tanh(train_df["v003"])
train_df['v007'] = train_df["ps_car_13"] + train_df["v002"] ** 2.7
train_df['v008'] = train_df["ps_car_13"] + train_df["v003"] ** 3.4
train_df['v009'] = train_df["ps_car_13"] + train_df["v004"] ** 3.1
train_df['v010'] = train_df["ps_car_13"] + train_df["v005"] ** 2.3

train_df['v011'] = train_df["ps_ind_03"] ** 2.1 + train_df["ps_ind_14"] ** 0.45 + train_df["ps_ind_15"] ** 2.4
train_df['v012'] = train_df["ps_ind_03"] ** 2.56 + train_df["ps_calc_13"] ** 2.15 + train_df["ps_reg_01"] ** 2.3
train_df['v013'] = train_df["v003"] ** 2.15 + train_df["ps_reg_01"] ** 2.49 + train_df["ps_ind_15"] ** 2.14
train_df['v014'] = train_df["v009"] ** 2.36 + train_df["ps_calc_01"] ** 2.25 + train_df["ps_reg_01"] ** 2.36
train_df['v015'] = train_df["v003"] ** 3.21 + 0.001 * np.tanh(train_df["ps_reg_01"]) + train_df["ps_ind_15"] ** 3.12
train_df['v016'] = train_df["v009"] ** 2.13 + 0.001 * np.tanh(train_df["ps_calc_01"]) + train_df["ps_reg_01"] ** 2.13
train_df['v017'] = train_df["v016"] ** 2 + train_df["v001"] ** 2.1 + train_df["v003"] ** 2.3

train_df['v018'] = train_df["v012"] ** 2.3 + train_df["v002"] ** 2.3 + train_df["v005"] ** 2.31
train_df['v019'] = train_df["v008"] ** 2.6 + train_df["v009"] ** 2.1 + train_df["v004"] ** 2.13
train_df['v020'] = train_df["v012"] ** 2.7 + train_df["v002"] ** 2.2 + train_df["v005"] ** 2.43

# test
test_df['v001'] = test_df["ps_ind_03"] + test_df["ps_ind_14"] + np.square(test_df["ps_ind_15"])
test_df['v002'] = test_df["ps_ind_03"] + test_df["ps_ind_14"] + np.tanh(test_df["ps_ind_15"])
test_df['v003'] = test_df["ps_reg_01"] + test_df["ps_reg_02"] ** 3 + test_df["ps_reg_03"]
test_df['v004'] = test_df["ps_reg_01"] ** 2.15 + np.tanh(test_df["ps_reg_02"]) + test_df["ps_reg_03"] ** 3.1
test_df['v005'] = test_df["ps_calc_01"] + test_df["ps_calc_13"] + np.tanh(test_df["ps_calc_14"])
test_df['v006'] = test_df["ps_car_13"] + np.tanh(test_df["v003"])
test_df['v007'] = test_df["ps_car_13"] + test_df["v002"] ** 2.7
test_df['v008'] = test_df["ps_car_13"] + test_df["v003"] ** 3.4
test_df['v009'] = test_df["ps_car_13"] + test_df["v004"] ** 3.1
test_df['v010'] = test_df["ps_car_13"] + test_df["v005"] ** 2.3

test_df['v011'] = test_df["ps_ind_03"] ** 2.1 + test_df["ps_ind_14"] ** 0.45 + test_df["ps_ind_15"] ** 2.4
test_df['v012'] = test_df["ps_ind_03"] ** 2.56 + test_df["ps_calc_13"] ** 2.15 + test_df["ps_reg_01"] ** 2.3
test_df['v013'] = test_df["v003"] ** 2.15 + test_df["ps_reg_01"] ** 2.49 + test_df["ps_ind_15"] ** 2.14
test_df['v014'] = test_df["v009"] ** 2.36 + test_df["ps_calc_01"] ** 2.25 + test_df["ps_reg_01"] ** 2.36
test_df['v015'] = test_df["v003"] ** 3.21 + 0.001 * np.tanh(test_df["ps_reg_01"]) + test_df["ps_ind_15"] ** 3.12
test_df['v016'] = test_df["v009"] ** 2.13 + 0.001 * np.tanh(test_df["ps_calc_01"]) + test_df["ps_reg_01"] ** 2.13
test_df['v017'] = test_df["v016"] ** 2 + test_df["v001"] ** 2.1 + test_df["v003"] ** 2.3

test_df['v018'] = test_df["v012"] ** 2.3 + test_df["v002"] ** 2.3 + test_df["v005"] ** 2.31
test_df['v019'] = test_df["v008"] ** 2.6 + test_df["v009"] ** 2.1 + test_df["v004"] ** 2.13
test_df['v020'] = test_df["v012"] ** 2.7 + test_df["v002"] ** 2.2 + test_df["v005"] ** 2.43

# ---- end FEATURE ENGINEERING: NONLINEAR feature engineering by Leandro dos Santos Coelho

# from olivier
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

    "v001", "v002", "v003", "v004", "v005",
    "v006", "v007", "v008", "v009", "v010",
    "v011", "v012", "v013", "v014", "v015",
    "v016", "v017", "v018", "v019", "v020",  # new nonlinear features
]

# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

# Process data
test_id = test_df['id'].values
train_id = train_df['id'].values
y = train_df['target']

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60))
    print('\r' * 75)
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + "_" + train_df[f2].apply(lambda x: str(x))
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + "_" + test_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))

    train_features.append(name1)

train = train_df[train_features]
test = test_df[train_features]

f_cats = [f for f in train.columns if "_cat" in f]

print (train.shape, test.shape, train.columns)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier
import sys

from basic_processing import *
from base_model import *
from target_encoding import *

model_set = {}

model_set['xgb1'] = {
    'name': 'xgb1',
    'mdl_type': 'xgb',
    'param': {
        'objective': 'binary:logistic',
        # 'booster' : 'gbtree',
        'learning_rate': 0.1,
        'n_estimators': 1000,

        'scale_pos_weight': 1.0,
        'seed': 1024,

        'colsample_bytree': 0.409,
        'gamma': 9.7,
        'max_depth': 5,
        'min_child_weight': 44,
        'subsample': 0.996,

        'random_state': 1024,
        'n_jobs': 7,
    },
    'metric': gini,
}

model_set['rgf1'] = {
    'name': 'rgf1',
    'mdl_type': 'rgf',
    'param': {
        'max_leaf': 290,
        'algorithm': "RGF",
        'loss':  "Log",
        'l2':  0.011,
        'sl2': 0.011,
        'normalize': False,
        'min_samples_leaf': 8,
        'n_iter':  None,
        'opt_interval': 100,
        'learning_rate': .4,
        'calc_prob': "sigmoid",
        'n_jobs': 7,
        'memory_policy': "generous",
        'verbose':  0,
    },
    'metric': gini,
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
    elif model['mdl_type'] == 'gbc':
        mdl = GradientBoostingClassifier(**model['param'])

    if model['mdl_type'] == 'xgb':
        early_stop = 50
    else:
        early_stop = 1

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging_with_target_encode(train, y,
                            test, train_id, test_id, metric, mdl, mdl_type=model['mdl_type'], seed=1024,
                            stratified=False, n_splits=1, n_folds=5, n_bags=1, early_stop=early_stop, verbose=False)
    write_csv(train_pred, test_pred, mdl_name=model['name']+'20nonlinear', mean=mean, overall_mean=full_score, std=std, bag=-1)
