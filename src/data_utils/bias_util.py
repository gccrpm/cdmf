from multiprocessing import Pool

import numpy as np
import pandas as pd

def genenrate_bias_csv(df, mode, pool_size):
    if mode == 'user':
        xname = 'userId'
    elif mode == 'movie':
        xname = 'movieId'
    else:
        raise ValueError('you must choose user or movie for mode.')
    mu = np.mean(df.loc[:, 'rating'])
    print(mu)

    all_xids = np.array(sorted(list(df.loc[:, xname].unique())), np.int32)

    global _generate_sub_bx_dict
    def _generate_sub_bx_dict(all_xids, df, start, end):
        sub_bx_dict = {}
        for i in range(start, end):
            xid = all_xids[i]
            sub_bx_dict[xid] = -mu + np.mean(
                df.loc[df[xname] == xid, 'rating'])
        return sub_bx_dict

    # multiprocessing
    pool = Pool(pool_size)
    res_list = []
    bx_dict = {}

    # split ids
    block_size = len(all_xids) // pool_size

    print('start generate {} bias dict'.format(mode))    
    for i in range(len(all_xids) // block_size + 1):
        start_idx = i * block_size
        end_idx = min(len(all_xids), (i + 1)*block_size)
        res = pool.apply_async(_generate_sub_bx_dict, args=(all_xids, df, start_idx, end_idx))
        res_list.append(res)
    pool.close()
    pool.join()

    # merge all sub_bu_dict
    for sub_bx_dict in [res.get() for res in res_list]:
        bx_dict.update(sub_bx_dict)
    print('{} bias generating have finished.'.format(mode))

    xids = []
    biases = []
    for xid, bias in bx_dict.items():
        xids.append(xid)
        biases.append(bias)
    
    pd.DataFrame({'{}Id'.format(mode):xids, 'bias':biases}).to_csv(
        '{}s_bias.csv'.format(mode), index=False)

    return None

if __name__ == '__main__':
    df_all_ratings = pd.read_csv(
        './using-data/cleaned_ratings.csv', 
        header=0,
        dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float, 'timestamp':np.str})
    genenrate_bias_csv(df_all_ratings, 'user', 8)
    genenrate_bias_csv(df_all_ratings, 'movie', 4)
    