import numpy as np
import pandas as pd

from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise.reader import Reader
from surprise.model_selection import train_test_split

data = 'ml-1m'

df_train_ratings = pd.read_csv('./using-data/{}/train_ratings.csv'.format(data), header=0,
                               dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float})
df_eval_ratings = pd.read_csv('./using-data/{}/eval_ratings.csv'.format(data), header=0,
                              dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float})

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(
    df_train_ratings.loc[:, ['userId', 'movieId', 'rating']], reader)
test_data = Dataset.load_from_df(
    df_eval_ratings.loc[:, ['userId', 'movieId', 'rating']], reader)

trainset = train_data.build_full_trainset()
_, testset = train_test_split(test_data, test_size=.999)

lu = [0.2, 0.02, 0.002]
lv = [0.2, 0.02, 0.002]

for u in lu:
    for v in lv:
        algo_nmf = NMF(n_factors=50, reg_pu=u, reg_qi=v)
        algo_nmf.fit(trainset)
        preds = algo_nmf.test(testset)
        rmse = accuracy.rmse(preds)

        with open('res_other_{}.txt'.format(data), 'a', encoding='utf-8') as f:
            f.write('nmf_{}_{}:{}'.format(str(u), str(v), str(rmse)))


