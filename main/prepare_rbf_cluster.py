import numpy as np


from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans

np.random.seed(1024)

gamma = 1.0

print ("Loading data...")

import numpy as np
import pandas as pd
import pickle

from basic_processing import *
from base_model import *

train, test, train_id, test_id, y = read_data()
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = drop_columns(train, col_to_drop)
test = drop_columns(test, col_to_drop)

cat_feature = [item for item in train.columns if item.endswith("cat")]
train, test = ohe(train, test, cat_features=cat_feature)

print (train.shape, test.shape)

print ("Combining data...")

all_data = pd.concat([train, test], axis=0).values
print ("combine data shape ", all_data.shape)

for n_clusters in [25, 50, 75, 100, 200]:
    part_name = 'cluster_rbf_%d' % n_clusters

    print ("Finding %d clusters..." % n_clusters)

    kmeans = MiniBatchKMeans(n_clusters, random_state=17 * n_clusters + 11, n_init=5)
    kmeans.fit(all_data)

    print ("Transforming data...")

    cluster_rbf = np.exp(- gamma * kmeans.transform(all_data))

    print ("Saving...")
    name = 'cluster_rbf_%d' % (n_clusters)

    with open(name+'train.pkl', 'wb') as f:
        pickle.dump(cluster_rbf[:train.shape[0]], f)
    with open(name + 'test.pkl', 'wb') as f:
        pickle.dump(cluster_rbf[train.shape[0]:], f)


print ("Done.")
