
import pdb
import os, inspect
import numpy as np
import torch


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data_path = os.path.join(self.data_path, 'raw')

        ''' VLSW parameter '''
        self.m = 2  # minimum length of the left input sequence
        self.n = 7  # maximum length of the left input sequence
        self.o = 2  # minimum length of the right input sequence
        self.p = 7  # maximum length of the right input sequence
        self.q = 5  # missing legnth

    def load_data(self, filename):
        return torch.load(os.path.join(self.data_path, filename))

    def VLSW(self, data, mulvar=True, data_idx=0):
        if not mulvar:  # mulvar은 나중에 생각해보자!
            data = data[data_idx, :]

        # packed_sequence 고려.
        # train_input = []
        # train_output = []
        # print('Left {} || Right {} || length {}'.format(0, 0, 0))
        # for len_left in range(self.m, self.n+1):
        #     for len_right in range(self.o, self.p+1):
        #         len_input = len_left + len_right + self.q
        #         print('Left {} || Right {} || length {}'.format(len_left, len_right, len_input))
        #         for idx in range(len(data) - len_input + 1):
        #             train_input.append(np.concatenate((data[idx:idx+len_left],
        #                                np.zeros(self.q),
        #                                data[idx+len_left+self.q:idx+len_input])))
        #             train_output.append(data[idx:idx+len_input])
        #             # train_output.append(data[idx+len_left:idx+len_left+self.q])
        #             # 이렇게 적혀있긴 한데, train data 마다마다 제각기 다른 길이를, model output에서 잘라낼 방법을 모르겠음...
        #             # 어차피 encoder-decoder 구조니까, 그냥 input == output 구조로 가자

        # return np.array(train_input), np.array(train_output)

        nb_train_data = 0
        for len_left in range(self.m, self.n+1):
            for len_right in range(self.o, self.p+1):
                len_input = len_left + len_right + self.q
                nb_train_data += (len(data) - len_input + 1)
        maximum_data_length = self.n + self.q + self.p

        train_input = np.zeros((nb_train_data, maximum_data_length))
        train_output = np.zeros((nb_train_data, maximum_data_length))
        train_lengths = np.zeros((nb_train_data))

        i = 0
        for len_left in range(self.m, self.n+1):
            for len_right in range(self.o, self.p+1):
                len_input = len_left + len_right + self.q
                print('Left {} || Right {} || length {}'.format(len_left, len_right, len_input))
                for idx in range(len(data) - len_input + 1):
                    train_input[i, :len_input] = np.concatenate((data[idx:idx+len_left],
                                                                 np.zeros(self.q),
                                                                 data[idx+len_left+self.q:idx+len_input]))
                    train_output[i, :len_input] = data[idx:idx+len_input]  # 일단 임의로 내가 input = output
                    train_lengths[i] = len_input
                    i += 1
        sorted_idx = np.argsort(train_lengths)[::-1]  # descending order

        train_lengths = train_lengths[sorted_idx]
        train_input = train_input[sorted_idx]
        train_output = train_output[sorted_idx]

        return train_input, train_output, train_lengths

    def _rs_inteldata(self, filename, filename_save):  # read and save
        data_folder = 'intellabdata'
        data_path = os.path.join(self.raw_data_path, data_folder)

        # pdb.set_trace()

        if filename == 'inteldata_44.npy':
            data = np.load(os.path.join(data_path, filename))  # (21, 48, 3)

            ''' Split train/ test data '''
            nb_train_data = 17  # days

            train_data = []
            test_data = []
            for i in range(data.shape[2]):
                train_data.append(data[:nb_train_data, :, i].ravel())
                test_data.append(data[nb_train_data:, :, i].ravel())
            train_data = np.array(train_data)
            test_data = np.array(test_data)

            ''' VLSW '''
            train_input, train_output = self.VLSW(train_data, mulvar=False, data_idx=0)
            # IL + M + IR / O

            # save numpy as .pt datafile
            torch.save(train_input, open(os.path.join(self.data_path, filename_save[0]), 'wb'))
            torch.save(train_output, open(os.path.join(self.data_path, filename_save[1]), 'wb'))

            ## test_data도 VLSW로 만들어둬야 할거 같은데...
            ## 우선 train이랑 validation만 해보자!

    def _rs_inteldata_packed(self, filename, filename_save):  # read and save
        data_folder = 'intellabdata'
        data_path = os.path.join(self.raw_data_path, data_folder)

        # pdb.set_trace()

        if filename == 'inteldata_44.npy':
            data = np.load(os.path.join(data_path, filename))  # (21, 48, 3)

            ''' Split train/ test data '''
            nb_train_data = 17  # days

            train_data = []
            test_data = []
            for i in range(data.shape[2]):
                train_data.append(data[:nb_train_data, :, i].ravel())
                test_data.append(data[nb_train_data:, :, i].ravel())
            train_data = np.array(train_data)
            test_data = np.array(test_data)

            ''' VLSW '''
            train_input, train_output, train_lengths = self.VLSW(train_data, mulvar=False, data_idx=0)
            # IL + M + IR / O

            # save numpy as .pt datafile
            torch.save(train_input, open(os.path.join(self.data_path, filename_save[0]), 'wb'))
            torch.save(train_output, open(os.path.join(self.data_path, filename_save[1]), 'wb'))
            torch.save(train_lengths, open(os.path.join(self.data_path, filename_save[2]), 'wb'))

if __name__ == '__main__':
    current_path = inspect.getfile(inspect.currentframe())  # 'Utils.py'
    current_dir = os.path.dirname(os.path.abspath(current_path))
    DATA_PATH = os.path.join(os.path.dirname(current_dir), 'data')

    # m, n, o, p, q = 2, 7, 2, 7, 5
    # m, n, o, p, q = 5, 5, 5, 5, 5

    dataUtil = Data(DATA_PATH)

    # data = dataUtil._rs_inteldata('inteldata_44.npy',
                            # ['ssim_tr_in_intel44.pt', 'ssim_tr_out_intel44.pt'])
    data = dataUtil._rs_inteldata_packed('inteldata_44.npy',
                            ['ssim_tr_in_pack_intel44.pt',
                             'ssim_tr_out_pack_intel44.pt',
                             'ssim_tr_len_pack_intel44.pt'])

    # pdb.set_trace()
