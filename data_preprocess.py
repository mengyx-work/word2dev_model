import cPickle as pickle
import os, collections
import pandas as pd


def generate_tensorboard_token_dict(pickle_file, token_dict_name='tensorboard_token_dict.tsv'):
    '''generate a .tsv file for TensorBoard visualization purpose
    '''
    pickle_file_path = os.path.join(os.path.expanduser("~"), pickle_file)
    with open(pickle_file_path, 'rb') as input_stream:
        data = pickle.load(input_stream)
    if 'reverse_token_dict' not in data.keys():
        raise ValueError("expected key not in pickle file {}".format(pickle_file_path))
    index = data['reverse_token_dict'].keys()
    token_name = [data['reverse_token_dict'][key] for key in index]
    token_data = pd.DataFrame({'index': index, 'token_name': token_name})
    output_pickle_file = os.path.join(os.path.expanduser("~"), token_dict_name)
    token_data.to_csv(output_pickle_file, sep='\t', index=False)
    print "finish creating TensorBoard dict with {} entries...".format(len(token_name))


def create_cbow_data(titles, context_window):
    '''create training_list and target_list from a list of titles, given the `context_window`.

    Return:
        training_list (list): a list of index sequence with fixed length 2*context_window for training
        target_list (list): a list of target
    '''
    span = 2 * context_window + 1
    missing_count = 0
    training_list = []
    target_list = []
    for title in titles:
        if len(title) < span:
            missing_count += 1
            continue
        buffer = collections.deque(maxlen=span)
        buffer.extend(title[:span])
        title_len = len(title)
        for i in xrange(title_len-span+1):
            buffer_list = list(buffer)
            target_list.append([buffer_list.pop(context_window)])
            training_list.append(buffer_list)
            if i + span < title_len:
                buffer.append(title[i+span])
    print '{} short titles are passed by context_window {}'.format(missing_count, context_window)
    return training_list, target_list


def generate_cbow_pickle_file(data_path, input_pickle_file, output_pickle_file, context_window):
    '''create the CBOW model training data(pickle file) from the processed title data (pickle file).
    '''
    with open(os.path.join(data_path, input_pickle_file)) as input_file:
        data = pickle.load(input_file)
    print 'found {} titles from pickle file'.format(len(data['titles']))
    training_list_, target_list_ = create_cbow_data(data['titles'], context_window)
    content = {'token_dict': data['token_dict'],
               'reverse_token_dict': data['reverse_token_dict'],
               'training_list': training_list_,
               'target_list': target_list_}
    with open(os.path.join(data_path, output_pickle_file), 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def CBOW_datq_generator():
    data_path = '/Users/matt.meng'
    #input_pickle_file = 'processed_titles_data.pkl'
    #output_pickle_file = 'titles_CBOW_data.pkl'
    input_pickle_file = 'lemmanized_no_stop_words_processed_titles.pkl'
    output_pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    context_window = 1
    generate_cbow_pickle_file(data_path, input_pickle_file, output_pickle_file, context_window)


def tensorboard_dict_generator():
    pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    generate_tensorboard_token_dict(pickle_file)

if __name__ == '__main__':
    CBOW_datq_generator()
