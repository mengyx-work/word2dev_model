import os, multiprocessing
from graph_model import word2vec
import tensorflow as tf
from model_utils import create_local_model_path, create_local_log_path


def predict_with_word2vec():
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
    max_X, min_X, mean_X = model.predict([2, 4, 5])
    print max_X
    print min_X


if __name__ == '__main__':
    NUM_THREADS = 2*multiprocessing.cpu_count()-1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')
    predict_with_word2vec()