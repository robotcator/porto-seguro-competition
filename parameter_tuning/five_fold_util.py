
import numpy as np
import pandas as pd


def five_fold(X, y, test, metric, mdl, train_id, test_id, seed=1024):
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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

        mdl = mdl.fit(X_train, y_train)
        valid_pred = mdl.predict_proba(X_valid)[:, 1]

        score = metric(y_valid, valid_pred)
        print ("score ", score)
        validation.append(score)

        train_temp_y[valid_index] = valid_pred
        test_pred_temp = mdl.predict_proba(test)[:, 1]
        test_temp_y += test_pred_temp

    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y / kfold
    print ("full score: ", metric(y, train_pred['pred_y']))

    return train_pred, test_pred


def five_fold_with_upsampling(X, y, test, metric, mdl, train_id, test_id, seed=1024, inc=1):
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    import numpy as np

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

        # Upsample during cross validation to avoid having the same samples
        # in both train and validation sets
        # Validation set is not up-sampled to monitor overfitting
        if inc:
            pos_train = X_train[y_train == 1]
            pos_y = y_train[y_train == 1]

            X_train = np.vstack([X_train, pos_train])
            y_train = np.concatenate([y_train, pos_y])

            idx = np.arange(len(X_train))
            np.random.shuffle(idx)

            X_train = X_train[idx]
            y_train = y_train[idx]

        mdl = mdl.fit(X_train, y_train)
        valid_pred = mdl.predict_proba(X_valid)[:, 1]

        score = metric(y_valid, valid_pred)
        print ("score ", score)
        validation.append(score)

        train_temp_y[valid_index] = valid_pred
        test_pred_temp = mdl.predict_proba(test)[:, 1]
        test_temp_y += test_pred_temp

    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y / kfold
    print ("score", metric(y, train_pred['pred_y']))

    return train_pred, test_pred


def five_fold_with_baging(X, y, test, metric, mdl, train_id, test_id, bag_num=5):
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.model_selection import StratifiedShuffleSplit

    import numpy as np

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id
    train_temp_y = np.zeros((bag_num, len(X)))

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id
    test_temp_y = np.zeros((bag_num, len(test)))

    random_seed = 1024
    for i in range(bag_num):
        print ("bagging num ", i + 1)
        seed = random_seed + i * 10

        kfold = 5
        skf = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True).split(X, y)

        validation = []
        for idx, (train_index, valid_index) in enumerate(skf):
            X_train, X_valid = X[train_index, :], X[valid_index, :]
            y_train, y_valid = y[train_index], y[valid_index]

            mdl = mdl.fit(X_train, y_train)
            valid_pred = mdl.predict_proba(X_valid)[:, 1]

            score = metric(y_valid, valid_pred)
            print ("score ", score)
            validation.append(score)

            train_temp_y[i, valid_index] = valid_pred
            test_pred_temp = mdl.predict_proba(test)[:, 1]
            test_temp_y[i, :] += test_pred_temp

        test_temp_y[i, :] /= kfold
        print ("mean: ", np.mean(validation))
        print ("std: ", np.std(validation))
        print ("score", metric(y, train_temp_y[i, :]))

    print (train_temp_y.shape, train_temp_y.mean(axis=0).shape)
    train_pred['pred_y'] = train_temp_y.mean(axis=0)
    test_pred['target'] = test_temp_y.mean(axis=0)

    print ("score", metric(y, train_pred['pred_y']))
    return train_pred, test_pred

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

def five_fold_new(X, y, test, metric, mdl, train_id, test_id, seed=1024):
    import pandas as pd

    kfold = 5

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id

    train_temp_y, test_temp_y = train(X, test, y, metric=metric, kfold=kfold)

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y
    print ("full score: ", metric(y, train_pred['pred_y']))

    return train_pred, test_pred