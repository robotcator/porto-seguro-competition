import pandas as pd
import numpy as np


def xgb_fit_predict(mdl, X_train, y_train, X_valid=None, y_valid=None,
                    test=None, eval_metric=['auc'], early_stop=10,
                    verbose=True, score_metric=None, seed=1024, valid=True):

    if valid == False:
        _eval_set = [(X_train, y_train)]
    else:
        _eval_set = [(X_train, y_train), (X_valid, y_valid)]

    mdl.seed = seed
    if early_stop > 0:
        mdl.fit(X_train, y_train, eval_set=_eval_set,
                eval_metric=eval_metric, early_stopping_rounds=early_stop, verbose=verbose)
    else:
        mdl.fit(X_train, y_train, eval_set=_eval_set, eval_metric=eval_metric, verbose=verbose)

    if early_stop > 0:
        print ("best iteration %d " % (mdl.best_iteration))
        # validation predict
        valid_pred = mdl.predict_proba(X_valid, ntree_limit=mdl.best_iteration+10)[:, 1]
        # test predict
        test_pred = mdl.predict_proba(test, ntree_limit=mdl.best_iteration+10)[:, 1]
    else:
        # validation predict
        valid_pred = mdl.predict_proba(X_valid)[:, 1]
        # test predict
        test_pred = mdl.predict_proba(test)[:, 1]

    score = score_metric(y_valid, valid_pred)
    return valid_pred, test_pred, score

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
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
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)

def five_fold_with_baging_with_target_encode(X, y, test, train_id, test_id, metric, mdl, mdl_type='xgb',
                          seed=1024, stratified=True, n_splits=1, n_folds=5, n_bags=1, early_stop=50,
                          verbose=True):
    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np
    import pandas as pd

    validation = []
    f_cats = [f for f in X.columns if "_cat" in f]

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id
    train_temp_y = np.zeros((len(X), n_splits * n_folds))

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id
    test_temp_y = np.zeros((len(test), n_splits * n_folds * n_bags))

    for nsplit in range(n_splits):
        # for split
        print ("Training split %d....." % nsplit)
        if stratified:
            skf = StratifiedKFold(n_splits=n_folds, random_state=seed + nsplit * 11, shuffle=True).split(X, y)
        else:
            skf = KFold(n_splits=n_folds, random_state=seed + nsplit * 11, shuffle=True).split(X, y)

        for fold, (train_index, valid_index) in enumerate(skf):
            # for fold
            print ("Training fold %d....." % fold)
            if (isinstance(X, pd.DataFrame)):
                X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[valid_index, :].copy()
                y_train, y_valid = y.iloc[train_index].copy(), y.iloc[valid_index]

            else:
                X_train, X_valid = X[train_index, :], X[valid_index, :]
                y_train, y_valid = y[train_index], y[valid_index]

            for f in f_cats:
                X_train[f + "_avg"], X_valid[f + "_avg"], test[f + "_avg"] = target_encode(
                    trn_series=X_train[f],
                    val_series=X_valid[f],
                    tst_series=test[f],
                    target=y_train,
                    min_samples_leaf=200,
                    smoothing=10,
                    noise_level=0
                )

            for bag in range(n_bags):
                # for bagging
                print ("Training bag %d....." % bag)
                random_seed = seed + 11 * nsplit + 17 * fold + 13 * bag
                # random_seed = seed + 11 * nsplit + 13 * bag
                # print ("seed: %d" % (random_seed))

                if mdl_type == 'xgb':
                    valid_pred, test_pred_temp, score = xgb_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=verbose, score_metric=metric, seed=random_seed,
                    )
                elif mdl_type == 'lgb':
                    valid_pred, test_pred_temp, score = lgb_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=verbose, score_metric=metric, seed=random_seed,
                    )
                elif mdl_type == 'rgf':
                    valid_pred, test_pred_temp, score = rgf_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=verbose, score_metric=metric, seed=random_seed
                    )
                elif mdl_type == 'lr':
                    valid_pred, test_pred_temp, score = lr_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=True, score_metric=metric, seed=random_seed
                    )

                print ("Fold %d-%d-%d score: %f" % (nsplit, fold, bag, score))

                validation.append(score)

                train_temp_y[valid_index, nsplit * n_bags + bag] = valid_pred
                test_temp_y[:, nsplit * n_folds * n_bags + fold * n_bags + bag] += test_pred_temp

    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    train_pred['pred_y'] = train_temp_y.mean(axis=1)
    test_pred['target'] = test_temp_y.mean(axis=1)
    full_score = metric(y, train_pred['pred_y'])

    print ("full score: ", full_score)
    return train_pred, test_pred, np.mean(validation), np.std(validation), full_score

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

def prepare_20nonlinear_feature():
    # Read data
    train_df = pd.read_csv('../input/train.csv', na_values="-1")  # .iloc[0:200,:]
    test_df = pd.read_csv('../input/test.csv', na_values="-1")

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
    train_df['v016'] = train_df["v009"] ** 2.13 + 0.001 * np.tanh(train_df["ps_calc_01"]) + train_df[
                                                                                                "ps_reg_01"] ** 2.13
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

    X = train_df[train_features]
    test = test_df[train_features]

    return X, test, train_id, test_id, y


# prepare data
# X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()

train, test, train_id, test_id, target = prepare_20nonlinear_feature()

print (train.shape, test.shape, train_id.shape, test_id.shape, target.shape)


# define the optimization
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

    train_pred, test_pred, mean, std, full_score = five_fold_with_baging_with_target_encode(train, target, test, train_id,
                                             test_id, metric, mdl, mdl_type='xgb', seed=1024, stratified=False,
                                             n_splits=1, n_folds=5, n_bags=1, early_stop=50, verbose=False)

    return mean

xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                            'colsample_bytree': (0.4, 1),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'min_child_weight': (1, 50),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)