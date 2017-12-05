def prepare_data_rp_entity_embedding_data():
    import pandas as pd
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # Preprocessing

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
import datetime
start = datetime.datetime.now()
X, test, train_id, test_id, y = prepare_data_rp_entity_embedding_data()
X = X.values
test = test.values
end = datetime.datetime.now()

print (X.shape, test.shape, (end-start).seconds)

# define the optimization
from bayes_opt import BayesianOptimization
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import auc, roc_auc_score, roc_curve
def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g

random_state = 1024
params = {
    'objective' : 'binary:logistic',
    # 'booster' : 'gbtree',
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'silent': 1,
    'scale_pos_weight': 1,
    'seed': random_state
}
metric = gini

# op_param = dict(colsample_bytree = 0.9663, gamma = 9.9114, max_depth = 5.4313, min_child_weight = 14.8164, subsample = 0.9936)
# params = dict(params, **op_param)

colsample_bytree = 0.9663
gamma = 9.9114
max_depth = 5.4313
min_child_weight = 14.8164
subsample = 0.9936

params['max_depth'] = int(max_depth)
params['gamma'] = max(gamma, 0)
params['min_child_weight'] = int(min_child_weight)
params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
params['subsample'] = max(min(subsample, 1), 0)

mdl = XGBClassifier(**params)

def plot_result(results, loss_name, save_dir, filename):
    # plot loss
    import matplotlib.pyplot as plt
    import os
    epochs = len(results['validation_0'][loss_name])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][loss_name], label='Train')
    ax.plot(x_axis, results['validation_1'][loss_name], label='Test')

    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    # plt.show()
    plt.savefig(os.path.join(save_dir, loss_name+filename))


def five_fold_for_xgb_withupsamle(X, y, test, metric, mdl, train_id, test_id, seed=1024, inc=1):
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

        # Upsample during cross validation to avoid having the same samples
        # in both train and validation sets
        # Validation set is not up-sampled to monitor overfitting
        # if inc:
        #     pos_train = X_train[y_train == 1]
        #     pos_y = y_train[y_train == 1]
        #
        #     X_train = np.vstack([X_train, pos_train])
        #     y_train = np.concatenate([y_train, pos_y])
        #
        #     idx = np.arange(len(X_train))
        #     np.random.shuffle(idx)
        #
        #     X_train = X_train[idx]
        #     y_train = y_train[idx]

        mdl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric=['auc'], early_stopping_rounds=50, verbose=False)

        result = mdl.evals_result()
        plot_result(result, 'auc', './', str(idx))

        valid_pred = mdl.predict_proba(X_valid, ntree_limit=mdl.best_iteration)[:, 1]
        score = metric(y_valid, valid_pred)
        print ("score ", score)
        validation.append(score)

        train_temp_y[valid_index] = valid_pred
        test_pred_temp = mdl.predict_proba(test, ntree_limit=mdl.best_iteration)[:, 1]
        test_temp_y += test_pred_temp

    print ("mean: ", np.mean(validation))
    print ("std: ", np.std(validation))

    train_pred['pred_y'] = train_temp_y
    test_pred['target'] = test_temp_y / kfold
    print ("full score: ", metric(y, train_pred['pred_y']))

    return train_pred, test_pred, np.mean(validation), np.std(validation)

def five_fold_with_baging(X, y, test, metric, mdl, train_id, test_id, bag_num=5):
    from sklearn.model_selection import StratifiedKFold

    import numpy as np
    import pandas as pd

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id
    train_temp_y = np.zeros((bag_num, len(X)))

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id
    test_temp_y = np.zeros((bag_num, len(test)))

    outter_validation = []
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

            mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=50, verbose=False)
            valid_pred = mdl.predict_proba(X_valid, ntree_limit=mdl.best_iteration)[:, 1]

            score = metric(y_valid, valid_pred)
            print ("score ", score)
            validation.append(score)
            outter_validation.append(score)

            train_temp_y[i, valid_index] = valid_pred
            test_pred_temp = mdl.predict_proba(test, ntree_limit=mdl.best_iteration)[:, 1]
            test_temp_y[i, :] += test_pred_temp

        test_temp_y[i, :] /= kfold
        print ("mean: ", np.mean(validation))
        print ("std: ", np.std(validation))
        print ("score", metric(y, train_temp_y[i, :]))

    print (train_temp_y.shape, train_temp_y.mean(axis=0).shape)
    train_pred['pred_y'] = train_temp_y.mean(axis=0)
    test_pred['target'] = test_temp_y.mean(axis=0)

    print ("score", metric(y, train_pred['pred_y']))
    return train_pred, test_pred, np.mean(outter_validation), np.std(outter_validation)

def five_fold_another_bagging(X, y, test, metric, mdl, train_id, test_id, bag_num=5):
    from sklearn.model_selection import StratifiedKFold

    import numpy as np
    import pandas as pd

    train_pred = pd.DataFrame()
    train_pred['id'] = train_id
    train_temp_y = np.zeros((bag_num, len(X)))

    test_pred = pd.DataFrame()
    test_pred['id'] = test_id
    test_temp_y = np.zeros((bag_num, len(test)))

    random_seed = 1024
    kfold = 5
    skf = StratifiedKFold(n_splits=kfold, random_state=random_seed, shuffle=True).split(X, y)

    outter_validation = []
    for idx, (train_index, valid_index) in enumerate(skf):
        X_train, X_valid = X[train_index, :], X[valid_index, :]
        y_train, y_valid = y[train_index], y[valid_index]
        validation = []

        for i in range(bag_num):
            print ("bagging num ", i + 1)
            mdl.seed = random_seed + i * 100
            # update the random seed
            mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', early_stopping_rounds=50, verbose=False)
            valid_pred = mdl.predict_proba(X_valid, ntree_limit=mdl.best_iteration)[:, 1]

            score = metric(y_valid, valid_pred)
            print ("score ", score)
            validation.append(score)
            outter_validation.append(score)

            train_temp_y[i, valid_index] = valid_pred
            test_pred_temp = mdl.predict_proba(test, ntree_limit=mdl.best_iteration)[:, 1]
            test_temp_y[i, :] += test_pred_temp

        print ("bagging mean: ", np.mean(validation))
        print ("bagging std: ", np.std(validation))

        train_temp_y[:, valid_index] /= bag_num

    test_temp_y /= kfold
    print (train_temp_y.shape, train_temp_y.mean(axis=0).shape)
    train_pred['pred_y'] = train_temp_y.mean(axis=0)
    test_pred['target'] = test_temp_y.mean(axis=0)

    print ("full score", metric(y, train_pred['pred_y']))
    return train_pred, test_pred, np.mean(outter_validation), np.std(outter_validation)


if __name__ == "__main__":

    # train_pred, test_pred, mean, std = five_fold_for_xgb_withupsamle(X, y, test, metric, mdl, train_id, test_id, 1024, inc=1)
    train_pred, test_pred, mean, std = five_fold_another_bagging(X, y, test, metric, mdl, train_id, test_id, bag_num=5)
    import datetime
    name = "%s-%s-%.5f-%.4f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), 'xgb', mean, std)

    # train_pred, test_pred = five_fold_with_baging(X, y, test, metric, mdl, train_id, test_id, bag_num=5)
    # temp = "5bag"
    # train_file_name = 'xgb-train-' + temp + '.csv'
    # test_file_name = 'xgb-test-' + temp + '.csv'

    train_pred.to_csv('%s-%s' % (name, 'train'), index=False)
    test_pred.to_csv('%s-%s' % (name, 'test'), index=False)

