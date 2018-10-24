import os
import re

import hyperparams as hp
from data_load import DataLoad
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

def load_ckpt_paths(model_name='cdmf'):
    # get ckpt
    ckpt_path = '../model_ckpt/{}/fine/'.format(model_name)
    fpaths = []
    with open(ckpt_path+'checkpoint', 'r', encoding='utf-8') as f_ckpt :
        for line in f_ckpt.readlines()[1:]:
            fname = re.sub(r'\"', '', line.split(':')[-1]).strip()
            fpath = os.path.join(ckpt_path, fname)
            fpaths.append(fpath)
    return fpaths

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            data = DataLoad(data_path=hp.DATA_PATH,
                            fnames=hp.FNAMES,
                            forced_seq_len=hp.FORCED_SEQ_LEN,
                            vocab_size=hp.VOCAB_SIZE,
                            paly_times=hp.PLAY_TIMES,
                            num_main_actors=hp.NUM_MAIN_ACTORS,
                            batch_size=hp.BATCH_SIZE,
                            num_epochs=hp.NUM_EPOCHS,
                            noise_rate=hp.NOISE_RATE)
            
            for fpath in load_ckpt_paths('cdmf'):
                saver = tf.train.import_meta_graph(fpath+'.meta')
                saver.restore(sess, fpath)

                # Get the placeholders from the graph by name
                m_oids = graph.get_tensor_by_name('movie_order_ids:0')
                info = graph.get_tensor_by_name('info:0')
                actors = graph.get_tensor_by_name('actors:0')
                descriptions = graph.get_tensor_by_name('descriptions:0')
                u_oids = graph.get_tensor_by_name('user_order_ids:0')
                r_seq = graph.get_tensor_by_name('rating_sequence:0')
                dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

                # Tensors we want to evaluate
                mse_op = graph.get_tensor_by_name('mse/mse_op:0')

                # load evalset
                eval_iter = data.load_data('eval')
                mse, count = 0.0, 0
                for (sub_X_user, sub_X_movie), sub_Y in tqdm(eval_iter):
                    # unpack
                    sub_u_oids, sub_bu_seq = sub_X_user
                    sub_m_oids, sub_info, sub_actors, sub_des, sub_bm_seq = sub_X_movie
                    sub_r_seq = sub_Y
                    dev_feed_dict = {
                        m_oids: sub_m_oids,
                        info: sub_info,
                        actors: sub_actors,
                        descriptions: sub_des,
                        u_oids: sub_u_oids,
                        r_seq: sub_r_seq,
                        dropout_keep_prob: hp.DROPOUT_KEEP_PROB}

                    sub_mse = sess.run(mse_op, feed_dict=dev_feed_dict)
                    mse += sub_mse
                    count += 1
                rmse = np.sqrt(mse / count)
                print('rmse:{}'.format(rmse))
