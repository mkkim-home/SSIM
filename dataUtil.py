
import pdb
import os, inspect
import numpy as np
import torch


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data_path = os.path.join(self.data_path, 'raw')

        ''' VLSW parameter '''
        # self.m = 2  # minimum length of the left input sequence
        # self.n = 7  # maximum length of the left input sequence
        # self.o = 2  # minimum length of the right input sequence
        # self.p = 7  # maximum length of the right input sequence
        # self.q = 5  # missing legnth

    def load_data(self, filename):
        return torch.load(os.path.join(self.data_path, filename))

    def preprocessing(self, input, target, lengths):
        print('DATA PREPROCESSING')

        print('\t1. NaN data check (deletion method)')
        nan_r_idx = np.unique(np.concatenate((np.where(np.isnan(input) == True)[0],
                                    np.where(np.isnan(target) == True)[0])))
        nan_r_idx[::-1].sort()  # for delete rows

        if len(nan_r_idx) > 0:
            print('\t\tWarning!!')
            print('\t\t{} rows with NaN data detected in train data'.format(nan_r_idx.shape[0]))

        for r_idx in nan_r_idx:
            input = np.delete(input, r_idx, axis=0)  # delete row
            target = np.delete(target, r_idx, axis=0)  # delete row
            lengths = np.delete(lengths, r_idx)
        print('\t\tNb. of train data: {} (NaN data are deleted)'.format(input.shape[0]))

        print('\t2. Convert data into torch.Tensor')
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        lengths = torch.from_numpy(lengths)

        print('\t3. Convert data into float (optional)')
        input = input.float()
        target = target.float()

        return input, target, lengths

    def VLSW(self, data, m, n, o, p, q, mulvar=True, data_idx=0):
        if not mulvar:
            data = data[data_idx, :]

        # draft of this function
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
        #             # train_output.append(data[idx+len_left:idx+len_left+self.q])  # only data at missing period
        #             # 논문에는 output의 size가 missing length와 동일하다고 되어있는데,
        #             # missing length 길이만큼만 decoding을 할때, 값의 시작을 무엇으로 해야할지 모르겠음...
        #             # 그래서, 어차피 encoder-decoder 구조니까, input == output으로 가보자
        # return np.array(train_input), np.array(train_output)

        nb_train_data = 0
        for len_left in range(m, n+1):
            for len_right in range(o, p+1):
                len_input = len_left + len_right + q
                nb_train_data += (len(data) - len_input + 1)
        maximum_data_length = n + q + p

        train_input = np.zeros((nb_train_data, maximum_data_length))
        train_output = np.zeros((nb_train_data, maximum_data_length))
        train_lengths = np.zeros((nb_train_data))

        i = 0
        for len_left in range(m, n+1):
            for len_right in range(o, p+1):
                len_input = len_left + len_right + q
                print('Left {} || Right {} || length {}'.format(len_left, len_right, len_input))
                for idx in range(len(data) - len_input + 1):
                    train_input[i, :len_input] = np.concatenate((data[idx:idx+len_left],
                                                                 np.zeros(q),
                                                                 data[idx+len_left+q:idx+len_input]))
                    train_output[i, :len_input] = data[idx:idx+len_input]  # ?????
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
            train_input, train_output = self.VLSW(train_data, m=2, n=7, o=2, p=7, q=5, mulvar=False, data_idx=0)
            # IL + M + IR / O

            # save numpy as .pt datafile
            torch.save(train_input, open(os.path.join(self.data_path, filename_save[0]), 'wb'))
            torch.save(train_output, open(os.path.join(self.data_path, filename_save[1]), 'wb'))


    def _rs_inteldata_packed(self, filename, tr_filename_save, te_filename_save):  # read and save
        data_folder = 'intellabdata'
        data_path = os.path.join(self.raw_data_path, data_folder)

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
            m = 2
            n = 7
            o = 2
            p = 7
            q = 5
            train_input, train_output, train_lengths = self.VLSW(train_data, m, n, o, p, q, mulvar=False, data_idx=0)
            # IL + M + IR / O

            # save numpy as .pt datafile
            torch.save(train_input, open(os.path.join(self.data_path, tr_filename_save[0]), 'wb'))
            torch.save(train_output, open(os.path.join(self.data_path, tr_filename_save[1]), 'wb'))
            torch.save(train_lengths, open(os.path.join(self.data_path, tr_filename_save[2]), 'wb'))


            # for test dataset,
            # make test dataset for every combination of length of left side and right side.
            for len_left in range(m, n+1):
                for len_right in range(o, p+1):
                    test_input, test_output, test_lengths = self.VLSW(test_data,
                    m=len_left, n=len_left, o=len_right, p=len_right, q=q, mulvar=False, data_idx=0)

                    torch.save(test_input, open(os.path.join(self.data_path, (te_filename_save[0] + '_{}_{}.pt'.format(len_left, len_right))), 'wb'))
                    torch.save(test_output, open(os.path.join(self.data_path, (te_filename_save[1] + '_{}_{}.pt'.format(len_left, len_right))), 'wb'))
                    torch.save(test_lengths, open(os.path.join(self.data_path, (te_filename_save[2] + '_{}_{}.pt'.format(len_left, len_right))), 'wb'))


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
                             'ssim_tr_len_pack_intel44.pt'],
                            ['ssim_te_in_pack_intel44',
                             'ssim_te_out_pack_intel44',
                             'ssim_te_len_pack_intel44'])


    # pdb.set_trace()
