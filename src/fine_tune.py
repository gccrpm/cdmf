import os
import re

import hyperparams as hp
from data_load import DataLoad
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

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
                            noise_rate=0)
            
            # get ckpt
            ckpt_path = '../model_ckpt/convmf/'
            with open(ckpt_path+'checkpoint', 'r', encoding='utf-8') as f_ckpt :
                fname = re.sub(r'\"', '', f_ckpt.readlines()[0].split(':')[-1]).strip()

            fpath = os.path.join(ckpt_path, fname)
            saver = tf.train.import_meta_graph(fpath+'.meta')
            saver.restore(sess, fpath)

            # Get the placeholders from the graph by name
            m_oids = graph.get_tensor_by_name('movie_order_ids:0')
            # info = graph.get_tensor_by_name('info:0')
            # actors = graph.get_tensor_by_name('actors:0')
            descriptions = graph.get_tensor_by_name('descriptions:0')
            u_oids = graph.get_tensor_by_name('user_order_ids:0')
            r_seq = graph.get_tensor_by_name('rating_sequence:0')
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

            # Get variables in model
            users_mat = graph.get_tensor_by_name('users_embedding/embedding_lookup:0')
            movies_epsilon = graph.get_tensor_by_name('movies_epsilon/embedding_lookup:0')
            square_errors = graph.get_tensor_by_name('Square:0')

            # Tensors we want to continue train
            # loss_op = graph.get_tensor_by_name('loss/loss_op:0')
            mse_op = graph.get_tensor_by_name('mse/mse_op:0')

            # create new operation
            with tf.variable_scope('tune_train'):
                loss_op = tf.reduce_sum(square_errors) + \
                    0.2 * tf.nn.l2_loss(movies_epsilon) + \
                    0.02 * tf.nn.l2_loss(users_mat)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                optimizer = tf.train.AdamOptimizer(0.0001)
                train_op = optimizer.minimize(
                    loss_op, global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()))
            sess.run(global_step.initializer)

            # saving dev evaluating reshults
            if not os.path.exists('../results/convmf/'):
                os.makedirs('../results/convmf/')
                with open('../results/convmf/tune_res.csv', 'a', encoding='utf-8') as resfile:
                    resfile.write('epoch,loss,rmse\n')

            saver_count = 0
            eval_count = 0
            last_best_rmse = 0.82

            for epoch in range(hp.NUM_EPOCHS):
                # load trainset
                train_batches = data.load_data('train')
                # training
                for (batch_X_user, batch_X_movie), batch_Y in tqdm(train_batches):
                    # unpack
                    batch_u_oids, batch_bu_seq = batch_X_user
                    batch_m_oids, batch_info, batch_actors, batch_des, batch_bm_seq = batch_X_movie
                    batch_r_seq = batch_Y
                    feed_dict = {
                        m_oids: batch_m_oids,
                        # info: batch_info,
                        # actors: batch_actors,
                        descriptions: batch_des,
                        u_oids: batch_u_oids,
                        r_seq: batch_r_seq,
                        dropout_keep_prob: hp.DROPOUT_KEEP_PROB}
                    _, step = sess.run(
                        [train_op, global_step],
                        feed_dict=feed_dict)

                    if step % hp.EVAL_EVERY == 0:
                        # load devset
                        dev_iter = data.load_data('dev')
                        # eval in devset
                        dev_loss, dev_mse, count = 0.0, 0.0, 0
                        for (sub_X_user, sub_X_movie), sub_Y in tqdm(dev_iter):
                            # unpack
                            sub_u_oids, sub_bu_seq = sub_X_user
                            sub_m_oids, sub_info, sub_actors, sub_des, sub_bm_seq = sub_X_movie
                            sub_r_seq = sub_Y
                            dev_feed_dict = {
                                m_oids: sub_m_oids,
                                # info: sub_info,
                                # actors: sub_actors,
                                descriptions: sub_des,
                                u_oids: sub_u_oids,
                                r_seq: sub_r_seq,
                                dropout_keep_prob: hp.DROPOUT_KEEP_PROB}

                            sub_loss, sub_mse = sess.run(
                                [loss_op, mse_op],
                                feed_dict=dev_feed_dict)
                            dev_loss += sub_loss
                            dev_mse += sub_mse
                            count += 1
                        dev_loss = dev_loss / count
                        dev_rmse = np.sqrt(dev_mse / count)
                        print('epoch:{} | loss:{} | rmse:{}'.format(epoch+1, dev_loss, dev_rmse))

                        # saving loss and rmse in devset
                        with open('../results/tune_res.csv', 'a', encoding='utf-8') as resfile:
                            resfile.write('{},{},{}\n'.format(epoch+1, dev_loss, dev_rmse))

                        # saving good model variables
                        if dev_rmse < 0.82 and dev_rmse > 0.80:
                            # last_best_rmse = dev_rmse
                            if not os.path.exists('../model_ckpt/convmf/fine/'):
                                os.makedirs('../model_ckpt/convmf/fine/')
                            saver.save(
                                sess, '../model_ckpt/convmf/fine/{}.ckpt'.format(dev_rmse))
