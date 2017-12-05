
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


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

def lgb_fit_predict(mdl, X_train, y_train, X_valid=None, y_valid=None,
                    test=None, eval_metric=['auc'], early_stop=10,
                    verbose=True, score_metric=None, seed=1024, valid=True):

    if valid == False:
        _eval_set = [(X_train, y_train)]
    else:
        _eval_set = [(X_train, y_train), (X_valid, y_valid)]

    mdl.set_params(**{'seed': seed})
    if early_stop > 0:
        mdl.fit(X_train, y_train, eval_set=_eval_set,
            eval_metric=eval_metric, early_stopping_rounds=early_stop, verbose=verbose)
    else:
        mdl.fit(X_train, y_train, eval_set=_eval_set, eval_metric=eval_metric, verbose=verbose)

    if early_stop > 0:
        # validation predict
        valid_pred = mdl.predict_proba(X_valid, num_iteration=mdl.best_iteration_)[:, 1]
        # test predict
        test_pred = mdl.predict_proba(test, num_iteration=mdl.best_iteration_)[:, 1]
    else:
        # validation predict
        valid_pred = mdl.predict_proba(X_valid)[:, 1]
        # test predict
        test_pred = mdl.predict_proba(test)[:, 1]

    score = score_metric(y_valid, valid_pred)

    return valid_pred, test_pred, score

def rgf_fit_predict(mdl, X_train, y_train, X_valid=None, y_valid=None,
                    test=None, eval_metric=['auc'], early_stop=10,
                    verbose=True, score_metric=None, seed=1024, valid=False):

    if X_valid is None:
        _eval_set = [(X_train, y_train)]
    else:
        _eval_set = [(X_train, y_train), (X_valid, y_valid)]

    # mdl.set_params(**{'seed': seed})
    mdl.fit(X_train, y_train)

    # validation predict
    valid_pred = mdl.predict_proba(X_valid)[:, 1]
    score = score_metric(y_valid, valid_pred)

    # test predict
    test_pred = mdl.predict_proba(test)[:, 1]

    try:
        import subprocess, glob
        subprocess.call('rm -rf /tmp/rgf/*', shell=True)
        print("Clean up is successfull")
        print(glob.glob("/tmp/rgf/*"))
    except Exception as e:
        print(str(e))

    return valid_pred, test_pred, score

def lr_fit_predict(mdl, X_train, y_train, X_valid=None, y_valid=None,
                    test=None, eval_metric=['auc'], early_stop=10,
                    verbose=True, score_metric=None, seed=1024, valid=False):

    mdl.random_state = seed
    mdl.fit(X_train, y_train)

    # validation predict
    valid_pred = mdl.predict_proba(X_valid)[:, 1]
    score = score_metric(y_valid, valid_pred)

    # test predict
    test_pred = mdl.predict_proba(test)[:, 1]

    return valid_pred, test_pred, score

def mlp_fit_predict(mdl, X_train, y_train, X_valid=None, y_valid=None,
                    test=None, eval_metric=['auc'], early_stop=10,
                    verbose=True, score_metric=None, seed=1024, valid=False):
    mdl.random_state = seed
    mdl.fit(X_train, y_train)

    # validation predict
    valid_pred = mdl.predict_proba(X_valid)[:, 1]
    score = score_metric(y_valid, valid_pred)

    # test predict
    test_pred = mdl.predict_proba(test)[:, 1]

    return valid_pred, test_pred, score



def five_fold(X, y, test, train_id, test_id, metric, mdl, mdl_type='xgb', seed=1024, early_stop=50):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import pandas as pd

    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True).split(X, y)

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id
    train_temp_y = np.zeros(len(X))

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id
    test_temp_y = np.zeros(len(test))

    validation = []

    for idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X[train_index, :], X[valid_index, :]
        y_train, y_valid = y[train_index], y[valid_index]

        random_seed = seed
        # print ("seed: %d" % (random_seed))
        if mdl_type == 'xgb':
            valid_pred, test_pred_temp, score = xgb_fit_predict(
                mdl, X_train, y_train, X_valid, y_valid,
                test=test, eval_metric=['auc'], early_stop=early_stop,
                verbose=False, score_metric=metric, seed=random_seed
            )
        elif mdl_type == 'lgb':
            valid_pred, test_pred_temp, score = lgb_fit_predict(
                mdl, X_train, y_train, X_valid, y_valid,
                test=test, eval_metric=['auc'], early_stop=early_stop,
                verbose=False, score_metric=metric, seed=random_seed,
            )
        elif mdl_type == 'rfg':
            valid_pred, test_pred_temp, score = rgf_fit_predict(
                mdl, X_train, y_train, X_valid, y_valid,
                test=test, eval_metric=['auc'], early_stop=early_stop,
                verbose=True, score_metric=metric, seed=random_seed
            )
        elif mdl_type == 'lr':
            valid_pred, test_pred_temp, score = lr_fit_predict(
                mdl, X_train, y_train, X_valid, y_valid,
                test=test, eval_metric=['auc'], early_stop=early_stop,
                verbose=True, score_metric=metric, seed=random_seed
            )
        print ("Fold %d score: %f" % (idx, score))

        validation.append(score)
        train_temp_y[valid_index] = valid_pred
        test_temp_y += test_pred_temp

    test_temp_y = test_temp_y / kfold
    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y
    full_score = metric(y, train_pred['pred_y'])

    print ("full score: ", full_score)

    return train_pred, test_pred, np.mean(validation), np.std(validation), full_score


def five_fold_with_baging(X, y, test, train_id, test_id, metric, mdl, mdl_type='xgb',
                          seed=1024, stratified=True, n_splits=1, n_folds=5, n_bags=1, early_stop=50,
                          verbose=True, inc=0):
    from sklearn.model_selection import StratifiedKFold, KFold
    import numpy as np
    import pandas as pd

    validation = []

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
            print ("using stratified kFold %d" % (seed + nsplit * 11))
            skf = StratifiedKFold(n_splits=n_folds, random_state=seed + nsplit * 11, shuffle=True).split(X, y)
        else:
            print ("using kFold %d" % (seed + nsplit * 11))
            skf = KFold(n_splits=n_folds, random_state=seed + nsplit * 11, shuffle=True).split(X, y)

        for fold, (train_index, valid_index) in enumerate(skf):
            # for fold
            print ("Training fold %d.....%d " % (fold, inc))

            if (isinstance(X, pd.DataFrame)):
                X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[valid_index, :].copy()
                y_train, y_valid = y.iloc[train_index].copy(), y.iloc[valid_index]

            else:
                X_train, X_valid = X[train_index, :], X[valid_index, :]
                y_train, y_valid = y[train_index], y[valid_index]

            if inc:
                print ("using upsample")
                pos_train = X_train[y_train == 1]
                pos_y = y_train[y_train == 1]

                X_train = np.vstack([X_train, pos_train])
                y_train = np.concatenate([y_train, pos_y])

                idx = np.arange(len(X_train))
                np.random.shuffle(idx)

                X_train = X_train[idx]
            #     y_train = y_train[idx]

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
                elif mdl_type == 'mlp':
                    valid_pred, test_pred_temp, score = mlp_fit_predict(
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

