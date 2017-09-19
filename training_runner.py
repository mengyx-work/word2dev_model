import os, multiprocessing
from graph_model import word2vec
import tensorflow as tf
from model_utils import create_local_model_path, create_local_log_path, generate_tensorboard_script
from data_feed import  DataGenerator


def model_train():

    #pickle_file = 'titles_CBOW_data.pkl'
    #pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    pickle_file = 'lemmanized_no_stop_words_CBOW_data_context_window_2.pkl'
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    dataGen = DataGenerator(pickle_file_path)

    model_config, training_config = {}, {}
    model_config['vocab_size'] = dataGen.vocab_size
    model_config['batch_size'] = 32
    model_config['context_window'] = 2
    model_config['embedding_size'] = 128
    model_config['neg_sample_size'] = 10
    model_config['learning_rate'] = 0.001
    model_config['saving_steps'] = 20000
    #model_config['model_name'] = 'word2vec_lemmatized_no_stop_words'
    model_config['model_name'] = 'word2vec_lemmatized_no_stop_words_context_window_2'
    model_config['restore_model'] = False
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
    training_config['display_steps'] = 10000
    training_config['saving_steps'] = model_config['saving_steps']
    training_config['num_batches'] = int(dataGen.data_size * epoch_num / model_config['batch_size'])
    print 'total #batches: {}, vocab_size: {}'.format(training_config['num_batches'], model_config['vocab_size'])
    model.train(**training_config)


if __name__ == '__main__':
    NUM_THREADS = 2*multiprocessing.cpu_count()-1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')
    model_train()
