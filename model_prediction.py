import cPickle as pickle
import os, multiprocessing, time
import tensorflow as tf
from graph_model import word2vec
from model_utils import create_local_model_path, create_local_log_path


def build_word2vec_model(model_name):
    ''' retrieve the model from persistent files.
    Args:
        model_name (string): the model name, assuming models
        are located in the standard folder.
    Return:
        model (word2vec object): the retrieved model
    '''
    NUM_THREADS = 2 * multiprocessing.cpu_count() - 1
    COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

    model_config = {}
    model_config['model_name'] = model_name
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
    return model


def collect_multi_keys_from_pickle_file(titles_pickle_file, key_dict):
    ''' collect useful lists from pickle file by the given `key_dict`.
    '''
    pickle_file_path = os.path.join(os.path.expanduser("~"), titles_pickle_file)
    with open(pickle_file_path, 'rb') as input_stream:
        data = pickle.load(input_stream)

    content_dict = {}
    for key in key_dict.keys():
        content_dict[key] = data[key_dict[key]]
    return content_dict


class ProcessedTitle(object):
    ''' class to represent individual processed title
    '''
    def __init__(self, index_title, url, pageView):
        self.index_title = index_title
        self.url = url
        self.pageView = pageView
        title_array = map(ProcessedTitle.reverse_token_dict.get, self.index_title)
        self.title = " ".join(title_array)

    def create_word2vec_embeddings(self, word2vec_model):
        max_vector, min_vector, mean_vector = word2vec_model.predict(self.index_title)
        self.max_vector = max_vector
        self.min_vector = min_vector
        self.mean_vector = mean_vector


def predict_titles_with_word2vec(content_dict, model, display_couner=1000, count_limit=None):
    processed_titles = []
    cur_time = time.time()
    if count_limit is None:
        count_limit = len(content_dict['titles'])
    else:
        count_limit = min(count_limit, len(content_dict['titles']))
    ProcessedTitle.reverse_token_dict = content_dict['reverse_token_dict']
    for i in xrange(count_limit):
        title = ProcessedTitle(index_title=content_dict['titles'][i],
                               url=content_dict['url'][i],
                               pageView=content_dict['pageView'][i])
        title.create_word2vec_embeddings(model)
        processed_titles.append(title)
        if i != 0 and i % display_couner == 0:
            print "processing {} titles using {:.2f} seconds".format(display_couner, time.time() - cur_time)
            cur_time = time.time()
    return processed_titles


def main():
    titles_pickle_file = 'lemmanized_no_stop_words_scrambled_titles.pkl'
    predicted_titles_pickle_file = 'predicted_titles.pkl'
    model_name = 'word2vec_lemmatized_no_stop_words'

    #expected_keys = {"titles": 'titles', "url": 'url', 'pageView': "pageViw", 'reverse_token_dict': 'reverse_token_dict'}
    lemmatized_expected_keys = {"titles": 'target_titles',
                                "url": 'url',
                                'pageView': "pageViw",
                                'reverse_token_dict': 'reverse_token_dict'}

    content_dict = collect_multi_keys_from_pickle_file(titles_pickle_file, lemmatized_expected_keys)
    word2vec_model = build_word2vec_model(model_name)
    processed_titles = predict_titles_with_word2vec(content_dict, word2vec_model, count_limit=2000)

    pickle_file_path = os.path.join(os.path.expanduser("~"), predicted_titles_pickle_file)
    with open(pickle_file_path, 'wb') as output_stream:
        pickle.dump(processed_titles, output_stream, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
