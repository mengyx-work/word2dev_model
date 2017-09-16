import cPickle as pickle
import os, collections

def create_cbow_data(titles, context_window):
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


def main():
    data_path = '/Users/matt.meng'
    input_pickle_file = 'processed_titles_data.pkl'
    output_pickle_file = 'titles_CBOW_data.pkl'

    with open(os.path.join(data_path, input_pickle_file)) as input_file:
        data = pickle.load(input_file)
    print 'found {} titles from pickle file'.format(len(data['titles']))
    training_list_, target_list_ = create_cbow_data(data['titles'], 2)
    content = {'token_dict': data['token_dict'],
               'reverse_token_dict': data['reverse_token_dict'],
               'training_list': training_list_,
               'target_list': target_list_}
    with open(os.path.join(data_path, output_pickle_file), 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()