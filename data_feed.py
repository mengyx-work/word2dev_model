import cPickle as pickle


class DataGenerator(object):

    def __init__(self, pickle_file_path):
        self._cur_index = 0
        with open(pickle_file_path, 'rb') as input_stream:
            data = pickle.load(input_stream)
        self.training_list = data['training_list']
        self.target_list = data['target_list']
        self.reverse_token_dict = data['reverse_token_dict']
        self.token_dict = data['token_dict']
        self.data_size = len(self.training_list)
        self.vocab_size = len(self.reverse_token_dict.keys())

    def generate_sequence(self, batch_size):
        if batch_size >= 2 * self.data_size:
            raise ValueError("the batch_size can not be more than two times the data_size")

        while True:
            if self._cur_index + batch_size <= self.data_size:
                start_index = self._cur_index
                self._cur_index += batch_size
                yield (self.training_list[start_index : self._cur_index], self.target_list[start_index : self._cur_index])
            else:
                start_index = self._cur_index
                self._cur_index = self._cur_index + batch_size - self.data_size
                training_batch_content = self.training_list[start_index : self.data_size]
                training_batch_content.extend(self.training_list[0 : self._cur_index])
                target_batch_content = self.target_list[start_index : self.data_size]
                target_batch_content.extend(self.target_list[0 : self._cur_index])
                yield (training_batch_content, target_batch_content)