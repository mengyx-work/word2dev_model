import tensorflow as tf
import numpy as np
import sys, math, collections


class word2vec(object):
    def __init__(self, config, model_type='CBOW'):
        self.model_type = model_type
        self.config = config

        # assert config.batch_size % config.num_skip == 0
        # assert config.num_skip <= 2 * config.context_window

    def _init_placeholders(self):
        if self.model_type == 'CBOW':
            self.X = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.context_window * 2],
                                    name="input_X")
        elif self.model_type == 'SKIP_GRAM':
            self.X = tf.placeholder(tf.int32, shape=[self.config.batch_size], name="input_X")
        else:
            raise ValueError('unknown model type {} is found...'.format(self.model_type))
        self.y = tf.placeholder(tf.int32, shape=[self.config.batch_size, 1], name="input_y")

    def _init_variables(self):
        init_width = 0.5 / self.config.embedding_size
        self.embedding = tf.Variable(
            tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -init_width, init_width),
            name='embedding')
        self.weight = tf.Variable(
            tf.truncated_normal([self.config.vocab_size, self.config.embedding_size],
                                stddev=1. / math.sqrt(self.config.embedding_size)),
            name='weight')
        self.bias = tf.Variable(tf.zeros([self.config.vocab_size]), name='bias')

    def cbow_batch_content(self):
        span = 2 * self.config.context_window + 1
        X = np.zeros(shape=(self.config.batch_size, span - 1), dtype=np.int32)
        y = np.zeros(shape=(self.config.batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)
        buffer.extend(np.random.randint(self.config.vocab_size, size=span))
        for i in xrange(self.config.batch_size):
            buffer_list = list(buffer)
            y[i, 0] = buffer_list.pop(self.config.context_window)
            X[i] = buffer_list
            buffer.append(np.random.randint(self.config.vocab_size, size=1))
        return X, y

    def _build_graph(self):
        X_embedded = tf.nn.embedding_lookup(self.embedding, self.X)
        if self.model_type == 'CBOW':
            X_embedded = tf.reduce_sum(X_embedded, 1)
        print 'shape: ', X_embedded
        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.weight,
                                                  self.bias,
                                                  inputs=X_embedded,
                                                  labels=self.y,
                                                  num_sampled=self.config.neg_sample_size,
                                                  num_classes=self.config.vocab_size))
        self.train = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

    def train(self):
        self._init_placeholders()
        self._init_variables()
        self._build_graph()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            step = 0
            while step < 50000:
                step += 1
                X, y = self.cbow_batch_content()
                _, loss = sess.run([self.train, self.loss], feed_dict={self.X: X,
                                                                       self.y: y})
                print 'the loss: {} at step {}'.format(loss, step)

