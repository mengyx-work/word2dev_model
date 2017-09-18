import tensorflow as tf
import numpy as np
import sys, os, math, collections, time, multiprocessing
from data import DataGenerator
from model_utils import create_local_model_path, create_local_log_path, clear_folder, \
    generate_tensorboard_script, model_meta_file


def single_variable_summary(var, name):
    reduce_mean = tf.reduce_mean(var)
    tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
    tf.summary.histogram('{}_histogram'.format(name), var)


class word2vec(object):
    """
    A word2dev model, supports both `CBOW` and `SKIP_GRAM` implementation.

    Attributes:
        vocab_size (int): the dimension of vocabulary
        embedding_size (int): the dimension of distributed representation of words
        embedding (TF Variable, [vocab_size, embedding_size]): the trainable word embedding

    Negative Contrastive Estimator (NCE) loss:
        neg_sample_size (int): the number of negative samples for each loss calculation
        weight (TF Variable, [vocab_size, embedding_size]): the NCE_Weight
        bias (TF Variable, [vocab_size]): the NCE_Bias
    """

    def __init__(self, sess_config, model_path, log_path, vocab_size=1024,
                 batch_size=32, context_window=2, embedding_size=512, neg_sample_size=2,
                 learning_rate=0.0001, saving_steps=1000, model_name='word2vec',
                 model_type='CBOW', eval_mode=False, restore_model=True):

        self.model_path = model_path
        self.log_path = log_path
        self.model_type = model_type
        self.batch_size = batch_size
        self.context_window = context_window
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.neg_sample_size = neg_sample_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.restore_model = restore_model

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        self.global_step = 0

        # assert config.batch_size % config.num_skip == 0
        # assert config.num_skip <= 2 * config.context_window

        if eval_mode:
            self._restore_model()
            self._eval_graph()
        elif not restore_model:
            clear_folder(self.log_path)
            clear_folder(self.model_path)
            self._build_graph(saving_steps)
        else:
            self._restore_model()


    def _init_placeholders(self):
        '''initialize the model placeholders, depending on the `model_type`.

        '''
        if self.model_type == 'CBOW':
            self.X = tf.placeholder(tf.int32, shape=[self.batch_size, self.context_window * 2],
                                    name="input_X")
        elif self.model_type == 'SKIP_GRAM':
            self.X = tf.placeholder(tf.int32, shape=[self.batch_size], name="input_X")
        else:
            raise ValueError('unknown model type {} is found...'.format(self.model_type))

        # placeholder for one title of varying length
        self.eval_X = tf.placeholder(tf.int32, shape=[None], name="eval_X")
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="input_y")

    def _init_variables(self, saving_steps):
        '''initialize the TF variables for model, add summary for them.
        '''
        self.global_saving_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_saving_step_op = tf.assign(self.global_saving_step,
                                                  self.global_saving_step + saving_steps,
                                                  name='increment_step')

        init_width = 0.5 / self.embedding_size
        self.embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -init_width, init_width),
            name='embedding')
        print "the embedding matrix: ", self.embedding

        self.weight = tf.Variable(
            tf.truncated_normal([self.vocab_size, self.embedding_size],
                                stddev=1. / math.sqrt(self.embedding_size)),
            name='weight')
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name='bias')

        tf.summary.histogram("embedding_matrix", self.embedding)
        tf.summary.histogram("NCE_weight", self.weight)
        tf.summary.histogram("NCE_bias", self.bias)

    def cbow_batch_content(self):
        '''generate a random set of X and y for CBOW model
        '''
        span = 2 * self.context_window + 1
        X = np.zeros(shape=(self.batch_size, span - 1), dtype=np.int32)
        y = np.zeros(shape=(self.batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)
        buffer.extend(np.random.randint(self.vocab_size, size=span))
        for i in xrange(self.batch_size):
            buffer_list = list(buffer)
            y[i, 0] = buffer_list.pop(self.context_window)
            X[i] = buffer_list
            buffer.append(np.random.randint(self.vocab_size, size=1))
        return X, y

    def _create_loss(self):
        '''build the graph, lookup the embedding for X and use NCE for loss

        For CBOW model:
            the input to `nce_loss` is the reduced sum of `X_embedded`.
        '''
        X_embedded = tf.nn.embedding_lookup(self.embedding, self.X)
        if self.model_type == 'CBOW':
            X_embedded = tf.reduce_sum(X_embedded, 1)
        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.weight,
                                                  self.bias,
                                                  inputs=X_embedded,
                                                  labels=self.y,
                                                  num_sampled=self.neg_sample_size,
                                                  num_classes=self.vocab_size), name="loss")
        single_variable_summary(self.loss, 'loss')
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name="train_op")


    def _eval_graph(self):
        '''create the eval graph
        '''
        with self.graph.as_default():
            self.norm_embedding = tf.nn.l2_normalize(self.embedding, 1)
            eval_X_embedded = tf.gather(self.norm_embedding, self.eval_X) # shape [title_len, embedding_size]
            self.max_eval_X_embedded = tf.reduce_max(eval_X_embedded, axis=0)
            self.min_eval_X_embedded = tf.reduce_min(eval_X_embedded, axis=0)
            self.mean_eval_X_embedded = tf.reduce_mean(eval_X_embedded, axis=0)


    def eval(self, title_list):
        with self.graph.as_default():
            max_X, min_X, mean_X = self.sess.run([self.max_eval_X_embedded,
                                                  self.min_eval_X_embedded,
                                                  self.mean_eval_X_embedded], feed_dict={self.eval_X: title_list})
        return max_X, min_X, mean_X


    def _retore_model_variables(self):
        '''restore the placeholders and Variables by name, given the session.
        '''
        self.X = self.sess.graph.get_tensor_by_name("input_X:0")
        self.y = self.sess.graph.get_tensor_by_name("input_y:0")
        self.eval_X = self.sess.graph.get_tensor_by_name("eval_X:0")

        self.global_saving_step = self.sess.graph.get_tensor_by_name("global_step:0")
        self.embedding = self.sess.graph.get_tensor_by_name("embedding:0")
        self.train_op = self.sess.graph.get_operation_by_name("train_op")
        self.loss = self.sess.graph.get_tensor_by_name("loss:0")
        self.increment_saving_step_op = self.sess.graph.get_tensor_by_name("increment_step:0")

    def _restore_model(self):
        '''restore the model from `self.model_path`, use `_retore_model_variables` to
        restore the variables and also restore the `self.global_step`
        '''
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model_meta_file(self.model_path))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))

            self._retore_model_variables()
            self.global_step = self.sess.run(self.global_saving_step)
        print 'restore trained models from {}'.format(self.model_path)
        print 'restore model from step: ', self.global_step

    def _build_graph(self, saving_steps):
        '''build the graph associated nodes and edges
        '''
        with self.graph.as_default():
            self._init_placeholders()
            self._init_variables(saving_steps)
            self._create_loss()
            self.saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
            self.sess.run(tf.global_variables_initializer())


    def saving_step_run(self):
        '''to run when `global_step` reaches the saving step point,
         save models and increment the global_step
        '''
        _ = self.sess.run(self.increment_saving_step_op)
        self.saver.save(self.sess, os.path.join(self.model_path, 'models'), global_step=self.global_saving_step)


    def display_step_run(self, start_time, training_batch, target_batch):
        '''to run when `global_step` reaches the display step point,
        display the loss and time cost
        Return:
            cur_time: the current time to update the time
        '''
        loss, summary = self.sess.run([self.loss, self.merged_summary_op],
                                      feed_dict={self.X: training_batch, self.y: target_batch})
        self.writer.add_summary(summary, self.global_step)
        cur_time = time.time()
        print 'the loss: {} at step {}, using {:.2f} minutes'.format(loss, self.global_step, 1.*(cur_time-start_time)/60.)
        return cur_time


    def train(self, batches, num_batches, saving_steps, display_steps):

        with self.graph.as_default():
            self.writer = tf.summary.FileWriter(self.log_path, self.graph)
            self.merged_summary_op = tf.summary.merge_all()

            start_time = time.time()
            while self.global_step < num_batches:
                training_batch, target_batch = next(batches)
                _ = self.sess.run([self.train_op], feed_dict={self.X: training_batch, self.y: target_batch})
                self.global_step += 1

                if self.global_step % saving_steps == 0:
                    self.saving_step_run()

                if self.global_step % display_steps == 0:
                    start_time = self.display_step_run(start_time, training_batch, target_batch)



def model_train():

    #pickle_file = 'titles_CBOW_data.pkl'
    pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path)

    model_config, training_config = {}, {}
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['batch_size'] = 32
    model_config['context_window'] = 1
    model_config['embedding_size'] = 128
    model_config['neg_sample_size'] = 2
    model_config['learning_rate'] = 0.0005
    model_config['saving_steps'] = 500
    model_config['model_name'] = 'word2vec'
    model_config['restore_model'] = True
    model_config['eval_mode'] = False
    batches = dataGen.generate_sequence(model_config['batch_size'])

    use_gpu = False
    if use_gpu:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])
    generate_tensorboard_script(model_config['log_path'])  # create the script to start a tensorboard session
    model = word2vec(**model_config)

    epoch_num = 20000
    training_config['batches'] = batches
    training_config['display_steps'] = 200
    training_config['saving_steps'] = model_config['saving_steps']
    training_config['num_batches'] = int(dataGen.data_size * epoch_num / model_config['batch_size'])
    print 'total #batches: {}, vocab_size: {}'.format(training_config['num_batches'], model_config['vocab_size'])
    model.train(**training_config)


def model_eval():
    model_config = {}
    model_config['model_name'] = 'word2vec'
    model_config['restore_model'] = True
    model_config['eval_mode'] = True

    use_gpu = False
    if use_gpu:
        model_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    model_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    model_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])
    model = word2vec(**model_config)
    max_X, min_X, mean_X = model.eval([2, 4, 5])
    print max_X
    print min_X


if __name__ == '__main__':
    NUM_THREADS = multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')
    model_eval()