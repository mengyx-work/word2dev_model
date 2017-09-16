import tensorflow as tf
import numpy as np
import sys, os, math, collections, time, multiprocessing
from data import DataGenerator
import stat


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


def generate_tensorboard_script(logdir):
    file_name = "start_tensorboard.sh"
    with open(file_name, "w") as text_file:
        text_file.write("#!/bin/bash \n")
        text_file.write("tensorboard --logdir={}".format(logdir))
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def model_meta_file(model_path, file_prefix="models"):
    meta_files = [f for f in os.listdir(model_path) if f[-5:] == '.meta']
    final_model_files = [f for f in meta_files if file_prefix in f]
    if len(final_model_files) == 0:
        raise ValueError("failed to find any model meta files in {}".format(model_path))
    if len(final_model_files) > 1:
        print "warning, more than one model meta file is found in {}".format(model_path)
    return os.path.join(model_path, final_model_files[0])


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
        return
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)

def single_variable_summary(var, name):
    reduce_mean = tf.reduce_mean(var)
    tf.summary.scalar('{}_reduce_mean'.format(name), reduce_mean)
    tf.summary.histogram('{}_histogram'.format(name), var)

class word2vec(object):
    def __init__(self, vocab_size, batch_size=32, context_window=2, embedding_size=512, neg_sample_size=2,
                 learning_rate=0.0001, model_name='word2vec', model_type='CBOW'):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.context_window = context_window
        self.embedding_size = embedding_size
        self.neg_sample_size = neg_sample_size
        self.learning_rate = learning_rate
        self.model_name = model_name

        # assert config.batch_size % config.num_skip == 0
        # assert config.num_skip <= 2 * config.context_window

    def _init_placeholders(self):
        if self.model_type == 'CBOW':
            self.X = tf.placeholder(tf.int32, shape=[self.batch_size, self.context_window * 2],
                                    name="input_X")
        elif self.model_type == 'SKIP_GRAM':
            self.X = tf.placeholder(tf.int32, shape=[self.batch_size], name="input_X")
        else:
            raise ValueError('unknown model type {} is found...'.format(self.model_type))
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="input_y")


    def _init_variables(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        init_width = 0.5 / self.embedding_size
        self.embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -init_width, init_width),
            name='embedding')
        self.weight = tf.Variable(
            tf.truncated_normal([self.vocab_size, self.embedding_size],
                                stddev=1. / math.sqrt(self.embedding_size)),
            name='weight')
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name='bias')

    def cbow_batch_content(self):
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

    def _build_graph(self):
        X_embedded = tf.nn.embedding_lookup(self.embedding, self.X)
        if self.model_type == 'CBOW':
            X_embedded = tf.reduce_sum(X_embedded, 1)
        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.weight,
                                                  self.bias,
                                                  inputs=X_embedded,
                                                  labels=self.y,
                                                  num_sampled=self.neg_sample_size,
                                                  num_classes=self.vocab_size))
        single_variable_summary(self.loss, 'loss')
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def _retore_model(self,sess):

        self.X = sess.graph.get_tensor_by_name("input_X:0")
        self.y = sess.graph.get_tensor_by_name("input_y:0")
        self.global_step = sess.graph.get_tensor_by_name("global_step:0")

        self.train = sess.graph.get_operation_by_name("Adam")
        self.loss = sess.graph.get_tensor_by_name("Mean:0")
        self.increment_global_step_op = sess.graph.get_tensor_by_name("increment_step:0")


    def train(self, batches, config, restore_model=False):
        if not restore_model:
            clear_folder(config['log_path'])
            clear_folder(config['model_path'])

            self._init_placeholders()
            self._init_variables()
            self._build_graph()

            self.increment_global_step_op = tf.assign(self.global_step, self.global_step + config['saving_steps'], name='increment_step')
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
        else:
            saver = tf.train.import_meta_graph(model_meta_file(config['model_path']))

        writer = tf.summary.FileWriter(config['log_path'])
        merged_summary_op = tf.summary.merge_all()

        with tf.Session(config=config['sess_config']) as sess:
            if not restore_model:
                writer.add_graph(sess.graph)
                sess.run(init)
                step = 0
            else:
                print 'restore trained models from {}'.format(config['model_path'])
                saver.restore(sess, tf.train.latest_checkpoint(config['model_path']))
                self._retore_model(sess)
                step = sess.run(self.global_step)
                print 'restore model from step: ', step

            start_time = time.time()
            while step < config['num_batches']:
                training_batch, target_batch = next(batches)
                _ = sess.run([self.train], feed_dict={self.X: training_batch, self.y: target_batch})
                step += 1

                if step % config['saving_steps'] == 0:
                    _ = sess.run(self.increment_global_step_op)
                    saver.save(sess, os.path.join(config['model_path'], 'models'), global_step=self.global_step)

                if step % config['display_steps'] == 0:
                    loss, summary = sess.run([self.loss, merged_summary_op], feed_dict={self.X: training_batch, self.y: target_batch})
                    writer.add_summary(summary, step)
                    cur_time = time.time()
                    print 'the loss: {} at step {}, using {:.2f} minutes'.format(loss, step, 1.*(cur_time-start_time)/60.)
                    start_time = cur_time



def main():

    NUM_THREADS = multiprocessing.cpu_count()
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    pickle_file = 'titles_CBOW_data.pkl'
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path)

    model_config, training_config = {}, {}
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['batch_size'] = 32
    model_config['context_window'] = 2
    model_config['embedding_size'] = 128
    model_config['neg_sample_size'] = 2
    model_config['learning_rate'] = 0.0005
    model_config['model_name'] = 'word2vec'
    batches = dataGen.generate_sequence(model_config['batch_size'])
    model = word2vec(**model_config)

    use_gpu = False
    if use_gpu:
        training_config['sess_config'] = tf.ConfigProto(log_device_placement=False,
                                                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        training_config['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    training_config['model_path'] = create_local_model_path(COMMON_PATH, model_config['model_name'])
    training_config['log_path'] = create_local_log_path(COMMON_PATH, model_config['model_name'])
    generate_tensorboard_script(training_config['log_path'])  # create the script to start a tensorboard session

    training_config['epoch_num'] = 20000
    training_config['display_steps'] = 10000
    training_config['saving_steps'] = 5 * training_config['display_steps']
    training_config['num_batches'] = int(dataGen.data_size * training_config['epoch_num'] / model_config['batch_size'])
    print 'total #batches: {}, vocab_size: {}'.format(training_config['num_batches'], model_config['vocab_size'])

    model.train(batches, training_config, restore_model=True)

if __name__ == '__main__':
    main()