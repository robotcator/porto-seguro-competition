import pandas as pd
import numpy as np

from base_model import *

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
                        mdl, X_train.values, y_train.values, X_valid.values, y_valid.values,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=verbose, score_metric=metric, seed=random_seed
                    )
                elif mdl_type == 'lr':
                    valid_pred, test_pred_temp, score = lr_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=True, score_metric=metric, seed=random_seed
                    )
                elif mdl_type == 'xgb-rankd':
                    valid_pred, test_pred_temp, score = xgb_fit_predict(
                        mdl, X_train, y_train, X_valid, y_valid,
                        test=test, eval_metric=['auc'], early_stop=early_stop,
                        verbose=verbose, score_metric=metric, seed=random_seed,
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