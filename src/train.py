import os

from models.cdmf import CDMF
from models.cdmf2 import CDMF2
from models.convmf import ConvMF
from data_load import DataLoad
import hyperparams as hp


from tqdm import tqdm
import numpy as np
import tensorflow as tf

def load_model(data, model_name):
    kwargs = {
        'num_all_users': data.num_all_users,
        'num_all_movies': data.num_all_movies,
        'num_all_info': data.num_all_info,
        'num_all_main_actors': data.num_all_main_actors,
        'vocab_size': data.vocab_size,
        'embedding_size': hp.EMBEDDING_SIZE,
        'feature_size': hp.FEATURE_SIZE,
        'forced_seq_len': hp.FORCED_SEQ_LEN,
        'num_most_info': data.num_most_info,
        'num_main_actors': data.num_main_actors,
        'dim_hidden1': hp.DIM_HIDDEN1,
        'dim_hidden2': hp.DIM_HIDDEN2,
        'dim_lantent': hp.DIM_LANTENT,
        'filters_size_list': hp.FILTERS_SIZE_LIST,
        'num_filters': hp.NUM_FILTERS,
        'l2_reg_lambda_u': hp.L2_REG_LAMBDA_U,
        'l2_reg_lambda_m': hp.L2_REG_LAMBDA_M,
        'l2_reg_lambda_cnn': hp.L2_REG_LAMBDA_CNN,
        'l2_reg_lambda_info1': hp.L2_REG_LAMBDA_INFO1,
        'l2_reg_lambda_info2': hp.L2_REG_LAMBDA_INFO2,
        'l2_reg_lambda_actors1': hp.L2_REG_LAMBDA_ACTORS1,
        'l2_reg_lambda_actors2': hp.L2_REG_LAMBDA_ACTORS2}
    if model_name.lower() == 'cdmf':
        model = CDMF(**kwargs)
    elif model_name.lower() == 'convmf':
        model = ConvMF(num_all_users=data.num_all_users,
                       num_all_movies=data.num_all_movies,
                       vocab_size=8000,
                       embedding_size=200,
                       forced_seq_len=hp.FORCED_SEQ_LEN,
                       dim_lantent=50,
                       filters_size_list=[3, 4, 5],
                       num_filters=100,
                       l2_reg_lambda_cnn=0.02,
                       l2_reg_lambda_u=0.02,
                       l2_reg_lambda_m=0.02)
    return model, '{}_{}_{}_{}'.format(model_name, hp.DATA, model.l2_reg_lambda_u, model.l2_reg_lambda_m)

if __name__ == '__main__':
    with tf.Graph().as_default():
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
                            num_epochs=hp.NUM_EPOCHS)

            model, model_name = load_model(data, 'cdmf')

            # define graph
            model.construct_netword()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=hp.LEARNING_RATE)
            train_op = optimizer.minimize(model.loss_op, global_step=global_step)

            # tensorboard graph visualizing
            graph_writer = tf.summary.FileWriter(logdir='../graph/{}/'.format(model_name))
            graph_writer.add_graph(sess.graph)

            # # Summaries for loss and rmse
            # loss_summary = tf.summary.scalar("loss", model.loss_op)
            # rmse_summary = tf.summary.scalar("rmse", model.rmse_op)

            # Train Summaries
            # train_summary_op = tf.summary.merge([loss_summary, rmse_summary])
            # train_summary_dir = os.path.join("../summaries/{}".format(model_name), "train")
            # train_summary_writer = tf.summary.FileWriter(
            #     train_summary_dir, sess.graph)

            # saving dev evaluating reshults
            if not os.path.exists('../results/{}/'.format(model_name)):
                os.makedirs('../results/{}/'.format(model_name))
            with open('../results/{}/res.csv'.format(model_name), 'a', encoding='utf-8') as resfile:
                resfile.write('step,loss,rmse\n')

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            bad_count = 0
            init_rmse = 0.86
            last_best_rmse = init_rmse
            last_best_loss = 0
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
                        model.m_oids:batch_m_oids,
                        model.info: batch_info,
                        model.actors: batch_actors,
                        model.descriptions: batch_des,
                        model.u_oids: batch_u_oids,
                        model.r_seq: batch_r_seq,
                        model.dropout_keep_prob: hp.DROPOUT_KEEP_PROB}
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
                                model.m_oids:sub_m_oids,
                                model.info: sub_info,
                                model.actors: sub_actors,
                                model.descriptions: sub_des,
                                model.u_oids: sub_u_oids,
                                model.r_seq: sub_r_seq,
                                model.dropout_keep_prob: hp.DROPOUT_KEEP_PROB}

                            sub_loss, sub_mse = sess.run(
                                [model.loss_op, model.mse_op],
                                feed_dict=dev_feed_dict)
                            dev_loss += sub_loss
                            dev_mse += sub_mse
                            count += 1
                        dev_loss = dev_loss / count
                        dev_rmse = np.sqrt(dev_mse / count)
                        print('step:{} | loss:{} | rmse:{}'.format(step, dev_loss, dev_rmse))

                        # saving loss and rmse in devset
                        with open('../results/{}/res.csv'.format(
                            model_name), 'a', encoding='utf-8') as resfile:
                            resfile.write('{},{},{}\n'.format(step, dev_loss, dev_rmse))

                        # saving good model variables
                        if dev_rmse < last_best_rmse:
                            last_best_rmse = dev_rmse
                            last_best_loss = dev_loss
                            if not os.path.exists('../model_ckpt/{}/'.format(model_name)):
                                os.makedirs('../model_ckpt/{}/'.format(model_name))
                            saver.save(
                                sess, '../model_ckpt/{}/{}.ckpt'.format(model_name, dev_rmse))
                        elif last_best_rmse < init_rmse and last_best_loss < dev_loss: 
                            # descreaing learning rate
                            bad_count += 1
                            with tf.variable_scope('train_{}'.format(bad_count)):
                                new_optimizer = tf.train.AdamOptimizer(
                                    learning_rate=hp.LEARNING_RATE / 2**bad_count)
                                train_op = new_optimizer.minimize(model.loss_op, global_step=global_step)
                                sess.run(tf.variables_initializer(new_optimizer.variables()))

