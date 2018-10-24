"""
nerual network
"""
import numpy as np
import tensorflow as tf


class ConvMF(object):
    def __init__(self, num_all_users, num_all_movies, vocab_size, embedding_size,  
                 forced_seq_len, dim_lantent, filters_size_list, num_filters, 
                 l2_reg_lambda_u, l2_reg_lambda_m, l2_reg_lambda_cnn):
        # embed params
        self.num_all_users = num_all_users
        self.num_all_movies = num_all_movies
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # input params
        self.forced_seq_len = forced_seq_len

        # model layer params
        self.dim_lantent = dim_lantent
        self.filters_size_list = filters_size_list
        self.num_filters = num_filters

        # regularization params
        self.l2_reg_lambda_u = l2_reg_lambda_u
        self.l2_reg_lambda_m = l2_reg_lambda_m
        self.l2_reg_lambda_cnn = l2_reg_lambda_cnn

        self.l2_loss_cnn = tf.constant(0.0, dtype=tf.float32)

        self._create_placeholder()

    def construct_netword(self):
        # lookup user vectors table
        users_mat = self._embed_users(self.u_oids)  # N*1*dim_lantent
        # lookup movie epsilon vectors table
        movies_epsilon = self._embed_movies_epsilon(
            self.m_oids)  # N*1*dim_lantent

        # extract text features
        embedded_expanded_sents = self._embed_word(
            self.descriptions)  # N*forced_seq_len*embedding_size
        droped_h = self._conv2d_banks(
            embedded_expanded_sents)  # N*total_num_filters
        description = self._dense_cnn1(droped_h)  # N*dim_lantent
        movies_mat = tf.expand_dims(self._dense_cnn2(
            description), 1)  # N*1*dim_lantent
        movies_mat = tf.add(movies_mat, movies_epsilon, name='movies_mat')

        # predict ratings
        r_seq_hat = self._rating_pred(users_mat, movies_mat)

        # compate squared error
        squared_errors = tf.square(tf.subtract(self.r_seq, r_seq_hat))

        # loss
        with tf.variable_scope('loss', reuse=None):
            reg_loss = self.l2_reg_lambda_u * tf.nn.l2_loss(users_mat) + \
                self.l2_reg_lambda_m * tf.nn.l2_loss(movies_epsilon) + \
                self.l2_reg_lambda_cnn * self.l2_loss_cnn
            self.loss_op = tf.add(
                reg_loss, tf.reduce_sum(squared_errors), name='loss_op')

        # mse
        with tf.variable_scope('mse', reuse=None):
            self.mse_op = tf.reduce_mean(squared_errors, name='mse_op')

        return self

    def _create_placeholder(self):
        self.m_oids = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name='movie_order_ids')
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
            for filter_size in self.filters_size_list:
                # pooled dimension is (N, 1, 1, num_filters)
                pooled = self._conv2d(inputs=inputs,
                                      filter_size=filter_size,
                                      padding='VALID')
                outputs.append(pooled)
            # combine all pooled features
            h_pool = tf.concat(outputs, axis=3)
            num_filters_total = self.num_filters * len(self.filters_size_list)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # dropout
            droped_h = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return droped_h  # N*num_filters_total

    def _conv2d(self, inputs, filter_size, scope='conv2d_maxpooling', padding='VALID', reuse=None):
        with tf.variable_scope(scope + '-{}'.format(filter_size), reuse=reuse):
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
            self.l2_loss_cnn += tf.nn.l2_loss(W)
            self.l2_loss_cnn += tf.nn.l2_loss(b)
        return pooled

    def _dense_cnn1(self, inputs, scope='dense_cnn1', reuse=None):
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
