import numpy as np
import pandas as pd

def prepare_data_rp_entity_embeding_data():
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

def prepare_data_back_encoder_data():
	# This Python 3 environment comes with many helpful analytics libraries installed
	# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
	# For example, here's several helpful packages to load in 

	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

	# Any results you write to the current directory are saved as output.
	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/test.csv')

	# Preprocessing 
	test_id = test['id'].values
	y = train['target'].values
	train_id = train['id'].values

	train = train.drop(['target','id'], axis = 1)
	test = test.drop(['id'], axis = 1)

	col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
	train = train.drop(col_to_drop, axis=1)  
	test = test.drop(col_to_drop, axis=1) 
	print ("drop calc feature ", train.shape, test.shape)

	cat_features = [a for a in train.columns if a.endswith('cat')]
	train = train.drop(cat_features, axis=1)  
	test = test.drop(cat_features, axis=1) 
	print ("drop cat feature ", train.shape, test.shape)

	train_ce_back_svd = pd.read_csv("train_ce_backencode_svd20.csv")
	test_ce_back_svd = pd.read_csv("test_ce_backencode_svd20.csv")
	train = pd.concat([train, train_ce_back_svd], axis=1)
	test = pd.concat([test, test_ce_back_svd], axis=1)
	print ("concat ce feature ", train.shape, test.shape)

	return train, test, y, train_id, test_id

####################################################################################
from bayes_opt import BayesianOptimization
import xgboost as xgb

train, test, y, train_id, test_id = prepare_data()
xgtrain = xgb.DMatrix(train, y)
##############################################
kfolds=5
num_rounds = 1000
random_state = 1024
num_iter = 25
init_points = 15

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):

	params['min_child_weight'] = int(min_child_weight)
	params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
	params['max_depth'] = int(max_depth)
	params['subsample'] = max(min(subsample, 1), 0)
	params['gamma'] = max(gamma, 0)
    
	xgbc = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=kfolds, stratified=True,
			 seed=random_state,
			 callbacks=[xgb.callback.early_stop(50)])

	val_score = xgbc['test-auc-mean'].values[-1]
	train_score = xgbc['train-auc-mean'].values[-1]

	return (2*val_score-1)

params = {
    'objective' : 'binary:logistic',
    'booster' : 'gbtree',
    'eta': 0.1,
    'silent': 1,
    'eval_metric': 'auc',
    'verbose_eval': True,
    'seed': random_state
}


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=kfolds, random_state=random_state, shuffle=True).split(train, y) 
# skf = KFold(n_splits=kfold, random_state=seed, shuffle=True).split(train, y)

xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                            'colsample_bytree': (0.4, 1),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'min_child_weight': (1, 20),
                                            })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)
