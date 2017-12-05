def train(X, test, y, metric, kfold, seed):
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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


def five_fold(X, y, test, metric, mdl, train_id, test_id, seed=1024):
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

def five_fold_bagging(X, y, test, metric, mdl, train_id, test_id, seed=1024):
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

def xgb_makenfold(nfold, seed, dall, param, fpreproc):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    sfk = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    idset = [x[1] for x in sfk.split(X=dall.get_label(), y=dall.get_label())]

    ret = []
    for k in range(nfold):
        dtrain = dall.slice(np.concatenate([idset[i] for i in range(nfold) if k != i]))
        dtest = dall.slice(idset[k])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param

        ret.append((dtrain, dtest))
    return ret