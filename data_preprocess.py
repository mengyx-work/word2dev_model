import cPickle as pickle
import os, collections, random
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


def create_skip_gram_training_data(titles, context_window, num_skips, shuffle_data=True):
    '''create training_list and target_list from a list of titles, given the `context_window`.

    Return:
        training_list (list): a list of index sequence with fixed length 2*context_window for training
        target_list (list): a list of target
    '''
    span = 2 * context_window + 1
    missing_count = 0
    training_list, target_list = [], []
    for title in titles:
        if len(title) < span:
            missing_count += 1
            continue
        word_buffer = collections.deque(maxlen=span)
        word_buffer.extend(title[:span])
        title_len = len(title)
        for i in xrange(title_len - span + 1):
            cur_target = word_buffer[context_window]
            targets_to_avoid = [cur_target]
            for j in xrange(num_skips):
                while cur_target in targets_to_avoid:
                    cur_target = random.randint(0, span - 1)
                targets_to_avoid.append(cur_target)
                target_list.append(cur_target)
                training_list.append(word_buffer[context_window])
            if i + span < title_len:
                word_buffer.append(title[i+span])

    if shuffle_data:
        indexes = range(len(target_list))
        random.shuffle(indexes)
        target_list = [target_list[i] for i in indexes]
        training_list = [training_list[i] for i in indexes]

    print 'skip gram model, {} short titles are passed by context_window {}'.format(missing_count, context_window)
    return training_list, target_list


def create_cbow_training_data(titles, context_window):
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
    print 'cbow model, {} short titles are passed by context_window {}'.format(missing_count, context_window)
    return training_list, target_list


def generate_word2vec_training_data(data_path, input_pickle_file, output_pickle_file,
                                    context_window, model_type="CBOW", num_skips=None):
    '''create the word2vec model training data(pickle file) from the processed title data (pickle file).
    '''
    with open(os.path.join(data_path, input_pickle_file)) as input_file:
        data = pickle.load(input_file)
    print 'found {} titles from pickle file'.format(len(data['titles']))

    if model_type == 'CBOW':
        training_list_, target_list_ = create_cbow_training_data(data['titles'], context_window)
    elif model_type == 'SKIP_GRAM':
        if num_skips is None:
            raise ValueError("the num_skips is missing...")
        training_list_, target_list_ = create_skip_gram_training_data(data['titles'], context_window, num_skips)
    else:
        raise ValueError("the model type {} is not recognized...".format(model_type))

    content = {'token_dict': data['token_dict'],
               'titles': data['titles'],
               'reverse_token_dict': data['reverse_token_dict'],
               'training_list': training_list_,
               'target_list': target_list_}

    with open(os.path.join(data_path, output_pickle_file), 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def word2vec_data_generator():
    ''' prepare the CBOW model training data from regular title pickle file
    '''
    config = {}
    config['data_path'] = '/Users/matt.meng'
    config['model_type'] = "SKIP_GRAM"
    config['context_window'] = 2
    config['num_skips'] = 2

    #input_pickle_file = 'processed_titles_data.pkl'
    #output_pickle_file = 'titles_CBOW_data.pkl'

    config['input_pickle_file'] = 'lemmatize_only_scrambled_1_times_titles.pkl'

    #output_pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    config['output_pickle_file'] = 'lemmatized_only_skip_gram_window_{}_skips_{}.pkl'.format(config['context_window'],
                                                                                             config['num_skips'])

    generate_word2vec_training_data(**config)


def tensorboard_dict_generator():
    #pickle_file = 'lemmanized_no_stop_words_CBOW_data.pkl'
    #token_dict_name = 'lemmatized_tensorboard_token_dict.tsv'

    #pickle_file = 'lemmatize_only_scrambled_1_times_titles.pkl'
    #token_dict_name = 'tensorboard_lemmatize_only_scrambled_dict.tsv'

    #pickle_file = 'update_lemmatize_only_scrambled_3_times_titles.pkl'
    #token_dict_name = 'tensorboard_update_lemmatize_only_scrambled_dict.tsv'

    pickle_file = 'full_dedup_scrambled_1_times_titles.pkl'
    token_dict_name = 'tensorboard_full_dedup_scrambled_1_dict.tsv'

    generate_tensorboard_token_dict(pickle_file, token_dict_name)

if __name__ == '__main__':
    #word2vec_data_generator()
    tensorboard_dict_generator()
