"""
nerual network
"""
import numpy as np
import tensorflow as tf


class CDMF2(object):
    def __init__(self, num_all_users, num_all_movies, num_all_info, num_all_main_actors,
                 vocab_size, embedding_size, feature_size, forced_seq_len, num_most_info,
                 num_main_actors, dim_hidden, dim_lantent, filters_size_list, num_filters,
                 l2_reg_lambda_u, l2_reg_lambda_m, l2_reg_lambda_cnn, l2_reg_lambda_info,
                 l2_reg_lambda_actors):
        # embed params
        self.num_all_users = num_all_users
        self.num_all_movies = num_all_movies
        self.num_all_info = num_all_info
        self.num_all_main_actors = num_all_main_actors
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # input params
        self.forced_seq_len = forced_seq_len
        self.num_most_info = num_most_info
        self.num_main_actors = num_main_actors

        # model layer params
        self.dim_hidden = dim_hidden
        self.dim_lantent = dim_lantent
        self.filters_size_list = filters_size_list
        self.num_filters = num_filters

        # regularization params
        self.l2_reg_lambda_u = l2_reg_lambda_u
        self.l2_reg_lambda_m = l2_reg_lambda_m
        self.l2_reg_lambda_cnn = l2_reg_lambda_cnn
        self.l2_reg_lambda_info = l2_reg_lambda_info
        self.l2_reg_lambda_actors = l2_reg_lambda_actors

        self._create_placeholder()

    def construct_netword(self):
        users_mat = self._embed_users(self.u_oids)  # N*1*dim_lantent
        movies_epsilon = self._embed_movies_epsilon(
            self.m_oids)  # N*1*dim_lantent

        # extract text features
        embedded_expanded_sents = self._embed_word(
            self.descriptions)  # N*forced_seq_len*embedding_size
        droped_h = self._conv2d_banks(
            embedded_expanded_sents)  # N*total_num_filters
        des_movies_mat = self._dense_cnn2(
            self._dense_cnn1(droped_h))  # N*dim_lantent

        # extract year/genres features
        # N*num_most_info*feature_size
        embedded_info = self._embed_info(self.info)
        embedded_info_flat = tf.reshape(
            embedded_info, [-1, self.num_most_info*self.feature_size])
        info_movies_mat = self._dense_info2(
            self._dense_info1(embedded_info_flat))  # N*dim_lantent

        # extract actors features
        embedded_actors = self._embed_actors(
            self.actors)  # N*num_main_actors*feature_size
        embedded_actors_flat = tf.reshape(
            embedded_actors, [-1, self.num_main_actors*self.feature_size])
        actors_movies_mat = self._dense_actors2(
            self._dense_actors1(embedded_actors_flat))  # N*dim_lantent

        # generate movie vecrors
        movies_mat = tf.add(info_movies_mat, actors_movies_mat)
        movies_mat = tf.expand_dims(
            tf.add(movies_mat, des_movies_mat), 1)  # N*1*dim_lantent
        movies_mat = tf.add(movies_mat, movies_epsilon, name='movies_mat')

        # predict ratings
        r_seq_hat = self._rating_pred(users_mat, movies_mat)

        # compate squared error
        squared_errors = tf.square(tf.subtract(self.r_seq, r_seq_hat))

        # loss
        with tf.variable_scope('loss', reuse=None):
            reg_loss = self.l2_reg_lambda_u * tf.nn.l2_loss(users_mat) + \
                self.l2_reg_lambda_m * tf.nn.l2_loss(movies_epsilon) + \
                self.l2_reg_lambda_cnn * self.l2_loss_cnn + \
                self.l2_reg_lambda_info * self.l2_loss_info + \
                self.l2_reg_lambda_actors * self.l2_loss_actors
            self.loss_op = tf.add(
                reg_loss, tf.reduce_sum(squared_errors), name='loss_op')

        # mse
        with tf.variable_scope('mse', reuse=None):
            self.mse_op = tf.reduce_mean(squared_errors, name='mse_op')

        # rmse
        with tf.variable_scope('rmse', reuse=None):
            self.rmse_op = tf.sqrt(self.mse_op, name='rmse_op')

        return self

    def _create_placeholder(self):
        self.m_oids = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name='movie_order_ids')
        self.info = tf.placeholder(
            dtype=tf.int32, shape=[None, self.num_most_info], name='info')
        self.actors = tf.placeholder(
            dtype=tf.int32, shape=[None, self.num_main_actors], name='actors')
        self.descriptions = tf.placeholder(
            dtype=tf.int32, shape=[None, self.forced_seq_len], name='descriptions')
        self.u_oids = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name='user_order_ids')
        self.r_seq = tf.placeholder(
            dtype=tf.float32, shape=[None], name='rating_sequence')
        self.dropout_keep_prob = tf.placeholder(
            dtype=tf.float32, name='dropout_keep_prob')
        return self

    def _embed_users(self, inputs, scope='users_embedding', reuse=None):
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table',
                                           dtype=tf.float32,
                                           shape=[self.num_all_users,
                                                  self.dim_lantent],
                                           initializer=tf.truncated_normal_initializer(
                                               mean=0.0, stddev=0.01))
            return tf.nn.embedding_lookup(lookup_table, inputs)

    def _embed_word(self, inputs, scope='word_embedding', reuse=None):
        """
        word embedding.
        """
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table',
                                           dtype=tf.float32,
                                           shape=[self.vocab_size,
                                                  self.embedding_size],
                                           initializer=tf.truncated_normal_initializer(
                                               mean=0.0, stddev=0.01))
            embedded_inputs = tf.nn.embedding_lookup(lookup_table, inputs)
        # dimension is (N, self.forced_seq_len, self.embedding_size, 1)
        return tf.expand_dims(embedded_inputs, -1)  # add channel

    def _conv2d_banks(self, inputs, scope='banks', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            outputs = []
            self.l2_loss_cnn = tf.constant(0, dtype=tf.float32)
            for filter_size in self.filters_size_list:
                # pooled dimension is (N, 1, 1, num_filters)
                pooled, sub_l2_loss_cnn = self._conv2d(inputs=inputs,
                                                       filter_size=filter_size,
                                                       padding='VALID')
                outputs.append(pooled)
                self.l2_loss_cnn += sub_l2_loss_cnn
            # combine all pooled features
            h_pool = tf.concat(outputs, axis=3)
            num_filters_total = self.num_filters * len(self.filters_size_list)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # dropout
            droped_h = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return droped_h  # N*num_filters_total

    def _conv2d(self, inputs, filter_size, scope='conv2d_maxpooling', padding='VALID', reuse=None):
        with tf.variable_scope(scope + '_{}'.format(filter_size), reuse=reuse):
            filter_shape = [filter_size,
                            self.embedding_size,
                            1,
                            self.num_filters]  # HWIO
            W = tf.get_variable(name='W',
                                shape=filter_shape,
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable(name='b',
                                shape=[self.num_filters],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            conv = tf.nn.conv2d(input=inputs,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding=padding,
                                name='conv')
            sub_l2_loss_cnn = tf.nn.l2_loss(W)
            sub_l2_loss_cnn += tf.nn.l2_loss(b)

            # nonlinear transition
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # maxpooling
            if padding == 'VALID':
                max_pool_size = self.forced_seq_len - filter_size + 1
            elif padding == 'SAME':
                max_pool_size = self.forced_seq_len + filter_size + 1
            pooled = tf.nn.max_pool(value=h,
                                    ksize=[1, max_pool_size, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding=padding,
                                    name='maxpool')
        return pooled, sub_l2_loss_cnn

    def _dense_cnn1(self, inputs, scope='dense_cnn1', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _dense_cnn2(self, inputs, scope='dense_cnn2', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _embed_info(self, inputs, scope='info_embedding', reuse=None):
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table',
                                           dtype=tf.float32,
                                           shape=[self.num_all_info,
                                                  self.feature_size],
                                           initializer=tf.truncated_normal_initializer(
                                               mean=0.0, stddev=0.01))
            return tf.nn.embedding_lookup(lookup_table, inputs)

    def _dense_info1(self, inputs, scope='dense_info1', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.l2_loss_info = tf.nn.l2_loss(W)
            self.l2_loss_info += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _dense_info2(self, inputs, scope='dense_info2', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.l2_loss_info += tf.nn.l2_loss(W)
            self.l2_loss_info += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _embed_actors(self, inputs, scope='actors_embedding', reuse=None):
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table',
                                           dtype=tf.float32,
                                           shape=[self.num_all_main_actors,
                                                  self.feature_size],
                                           initializer=tf.truncated_normal_initializer(
                                               mean=0.0, stddev=0.01))
            return tf.nn.embedding_lookup(lookup_table, inputs)

    def _dense_actors1(self, inputs, scope='dense_actors1', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_hidden],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.l2_loss_actors = tf.nn.l2_loss(W)
            self.l2_loss_actors += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _dense_actors2(self, inputs, scope='dense_actors2', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1],
                                       self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.dim_lantent],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.l2_loss_actors += tf.nn.l2_loss(W)
            self.l2_loss_actors += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(inputs, W, b, name='logits')
            outputs = tf.nn.tanh(logits, name='outputs')
        return outputs

    def _embed_movies_epsilon(self, inputs, scope='movies_epsilon', reuse=None):
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='movies_epsilon',
                                           dtype=tf.float32,
                                           shape=[self.num_all_movies,
                                                  self.dim_lantent],
                                           initializer=tf.truncated_normal_initializer(
                                                mean=0.0, stddev=0.01))
            return tf.nn.embedding_lookup(lookup_table, inputs)

    def _rating_pred(self, users_mat, movies_mat):
        def _rating_pred_helper(x):
            user_vec = tf.reshape(x[0], (self.dim_lantent,))
            movie_vec = tf.reshape(x[1], (self.dim_lantent,))
            return tf.reduce_sum(tf.multiply(user_vec, movie_vec))

        r_seq_hat = tf.map_fn(_rating_pred_helper,
                              [users_mat, movies_mat],
                              dtype=tf.float32)
        return r_seq_hat
