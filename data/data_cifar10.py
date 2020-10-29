import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from data.data_preprocessing import *

def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

dir_10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir_100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"

class Cifar10() :
    def __init__(self, directory=None, step = None, number_label = None):
        self.directory = directory
        self.step = step
        self.number_label = number_label
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)
        self.ordered_test_X, self.ordered_test_Y = self.init_ordered_testset(self.test_X, self.test_Y)
        self.split_dataset= self._split_by_label(10, self.train_X, self.train_Y)
        self.split_testset = self._split_by_label(10, self.test_X, self.test_Y)

    def load_cifar10_batch(self, directory):
        """ load batch of cifar """
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'labels'])
        return X, Y

    def load_cifar10(self, directory):
        """ load all of cifar10 """
        train_data = []
        train_labels = []
        for b in range(1, 6):
            f = os.path.join(directory, 'data_batch_%d' % (b,))
            X, Y = self.load_cifar10_batch(f)
            train_data.append(X)
            train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar10_batch(os.path.join(directory, 'test_batch'))
        return train_data, train_labels, test_data, test_labels

    def load_cifar10_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar10 batch 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar10(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''

        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar10_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def init_ordered_testset(self, data_X, data_Y):

        temp_list = self._split_by_label(10, data_X, data_Y)
        temp_X = []
        temp_Y = []
        for i in range(temp_list.__len__()) :
            temp_X.append(temp_list[i][0])
            temp_Y.append(temp_list[i][1])

        test_X = np.concatenate(temp_X, axis = 0)
        test_Y = np.concatenate(temp_Y, axis = 0)

        return test_X, test_Y

    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

    def make_subset(self, split_dataset, label) :
        '''
        :param split_dataset: dataset spliited by each label
        :param label: label which is involved in trainset
        :return:
        '''
        X_list = []
        Y_list = []
        print(len(split_dataset))
        for i in range(len(split_dataset)) :
            if i in label :
                X_list.append(split_dataset[i][0])
                Y_list.append(split_dataset[i][1])

        data_list = [X_list, Y_list]
        return data_list

    def make_test_subset(self, split_dataset, label):
        '''
                :param split_dataset: dataset spliited by each label
                :param label: label which is involved in testset
                :return: data subset
        '''

        X_list = []
        Y_list = []

        print(len(split_dataset))
        for i in range(len(split_dataset)):
            if i in label:
                X_list.append(split_dataset[i][0])
                Y_list.append(split_dataset[i][1])

        data_list = [X_list, Y_list]

        return data_list

    def weighted_sampling(self, class_weight = None, split_dataset = None, class_min = None, share_memory=None):
        '''
        :param class_weight: 클래스별 데이터 샘플링 가중치 0~1
        :param split_dataset: label별로 나누어진 데이터 셋
        :param class_min: 각 클래스별 필요한 최소 데이터
        :param share_memory: 가중치 샘플링을 적용할 공유 메모리
        :return: new weighted data
        '''
        temp_data_X = []
        temp_data_Y = []

        class_data_index = (np.array(class_weight) * share_memory) + class_min
        print(class_data_index)
        #print(class_data_index)
        #class_data_index.astype(np.int)
        num_class = len(class_weight)

        for i in range(num_class) :
            temp_index = np.random.choice(np.arange(0, int(len(split_dataset[i][1]))), int(class_data_index[i]), replace = False)
            temp_data_X.append(split_dataset[i][0][temp_index])
            temp_data_Y.append(split_dataset[i][1][temp_index])

        weighted_data_X = np.concatenate(temp_data_X, axis=0)
        weighted_data_Y = np.concatenate(temp_data_Y, axis=0)

        weighted_data = [weighted_data_X, weighted_data_Y]

        return weighted_data


class imbalanced_Cifar10() :
    def __init__(self, directory=None, minority_label = None, majority_label=None, imbalance_ratio = None):
        self.directory = directory
        self.minority_label = minority_label
        self.majority_label = majority_label
        self.imbalance_ratio = imbalance_ratio

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)

        self.split_dataset= self._split_by_label(10, self.train_X, self.train_Y)
        self.split_testset = self._split_by_label(10, self.test_X, self.test_Y)

        self.imbalance_data_set = self.make_imbalance_dataset(split_dataset=self.split_dataset,
                                                               majority_label=self.majority_label,
                                                               minority_label=self.minority_label)

        #self.subset_data = self.make_subset(split_dataset=self.seplit_dataset, label = None)

        self.valid_ratio = 1

        self.major_validation_index, self.minor_validation_index = self.set_valid_index(split_dataset = self.split_dataset, majority_label=self.majority_label,
                                                               minority_label = self.minority_label, valid_ratio=self.valid_ratio)

        '''split the trainset into two part for the retraining'''
        self.imbalance_train_set = self.make_imbalance_trainset(imbalance_dataset = self.imbalance_data_set,
                                                                majority_label=self.majority_label, minority_label=self.minority_label)

        self.validation_set = self.make_valid_set(split_dataset = self.split_dataset, majority_label=self.majority_label,
                                                               minority_label = self.minority_label)



    def load_cifar10_batch(self, directory):
        """ load batch of cifar """
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'labels'])
        return X, Y

    def load_cifar10(self, directory):
        """ load all of cifar10 """
        train_data = []
        train_labels = []
        for b in range(1, 6):
            f = os.path.join(directory, 'data_batch_%d' % (b,))
            X, Y = self.load_cifar10_batch(f)
            train_data.append(X)
            train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar10_batch(os.path.join(directory, 'test_batch'))
        return train_data, train_labels, test_data, test_labels

    def load_cifar10_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar10 batch 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar10(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''

        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar10_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

    def make_subset(self, split_dataset, label) :
        X_list = []
        Y_list = []
        print(len(split_dataset))
        for i in range(len(split_dataset)) :
            if i in label :
                X_list.append(split_dataset[i][0])
                Y_list.append(split_dataset[i][1])

        data_list = [X_list, Y_list]
        return data_list

    def make_test_subset(self, split_dataset, label):
        X_list = []
        Y_list = []

        print(len(split_dataset))
        for i in range(len(split_dataset)):
            if i in label:
                X_list.append(split_dataset[i][0])
                Y_list.append(split_dataset[i][1])

        data_list = [X_list, Y_list]

        return data_list

    def make_imbalance_dataset(self, split_dataset, majority_label, minority_label):
        '''

        :param split_dataset: dataset which is spliited by label
        :return: imbalanced dataset(dtype : list)
        '''

        temp_data_X = []
        temp_data_Y = []
        for i in majority_label :
            if i == 10 :
                break
            #temp_label = np.where(split_dataset[current_step][1] == i)
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[i][0])
            temp_data_Y.append(split_dataset[i][1])

        if i==10 :
            majority_data_X = None
            majority_data_Y = None
        else :
            majority_data_X = np.concatenate(temp_data_X, axis=0)
            majority_data_Y = np.concatenate(temp_data_Y, axis=0)


    #    del temp_data_X, temp_data_Y
    #    temp_data_X = []
    #    temp_data_Y = []

        for j in minority_label :
            temp_index = np.random.choice(np.arange(0, len(split_dataset[j][1])), int(len(split_dataset[j][1])/self.imbalance_ratio),
                                          replace=False)

            #np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[j][0][temp_index])
            temp_data_Y.append(split_dataset[j][1][temp_index])

        minority_data_X = np.concatenate(temp_data_X, axis=0)
        minority_data_Y = np.concatenate(temp_data_Y, axis=0)

        if i==10 :
            imbalanced_data_X = minority_data_X
            imbalanced_data_Y = minority_data_Y
        else :
            imbalanced_data_X = np.concatenate((minority_data_X, majority_data_X), axis=0)
            imbalanced_data_Y = np.concatenate((minority_data_Y, majority_data_Y), axis=0)

        imbalanced_dataset =[temp_data_X, temp_data_Y]

        return imbalanced_dataset


    def set_valid_index(self,split_dataset, majority_label, minority_label, valid_ratio):
        '''make validation for second phase training set'''
        major_valid_index = np.random.choice(np.arange(0, len(split_dataset[0][1])),
                                             int(len(split_dataset[0][1]) / self.imbalance_ratio * valid_ratio),
                                             replace=False)

        minor_valid_index = np.random.choice(np.arange(0, int(len(split_dataset[0][1])/self.imbalance_ratio)),
                                             int(len(split_dataset[0][1])/self.imbalance_ratio * valid_ratio),
                                          replace=False)

        return major_valid_index, minor_valid_index

    def make_valid_set(self, split_dataset, majority_label, minority_label):
        temp_data_X = []
        temp_data_Y = []

        for i in majority_label :
            if i == 10 :
                break
            #temp_label = np.where(split_dataset[current_step][1] == i)
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[i][0][self.major_validation_index])
            temp_data_Y.append(split_dataset[i][1][self.major_validation_index])

        if i==10 :
            majority_val_data_X = None
            majority_val_data_Y = None
        else :
            majority_val_data_X = np.concatenate(temp_data_X, axis=0)
            majority_val_data_Y = np.concatenate(temp_data_Y, axis=0)

        del temp_data_X, temp_data_Y

        temp_data_X = []
        temp_data_Y = []

        for j in minority_label :

            #np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[j][0][self.minor_validation_index])
            temp_data_Y.append(split_dataset[j][1][self.minor_validation_index])

        minority_val_data_X = np.concatenate(temp_data_X, axis=0)
        minority_val_data_Y = np.concatenate(temp_data_Y, axis=0)

        if i==10 :
            imbalanced_data_X = minority_val_data_X
            imbalanced_data_Y = minority_val_data_Y
        else :
            imbalanced_data_X = np.concatenate((minority_val_data_X, majority_val_data_X), axis=0)
            imbalanced_data_Y = np.concatenate((minority_val_data_Y, majority_val_data_Y), axis=0)

        validation_dataset =[imbalanced_data_X, imbalanced_data_Y]

        np.save("./checkpoint/val_X", imbalanced_data_X)
        np.save("./checkpoint/val_Y", imbalanced_data_Y)
        del temp_data_X, temp_data_Y

        return validation_dataset


    def make_imbalance_trainset(self, imbalance_dataset, majority_label, minority_label) :
        '''make trainset for first phase training set'''
        #except validation set
        temp_data_X = []
        temp_data_Y = []

        for i in majority_label:
            if i == 10:
                break
            # temp_label = np.where(split_dataset[current_step][1] == i)
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태
            print(i)
            a = np.arange(0,len(imbalance_dataset[0][i]))
            temp_set = np.delete(a, self.major_validation_index)

            print(temp_set.__len__())
            temp_data_X.append(imbalance_dataset[0][i][temp_set])
            temp_data_Y.append(imbalance_dataset[1][i][temp_set])

        if i == 10:
            majority_tr_data_X = None
            majority_tr_data_Y = None
        else:
            majority_tr_data_X = np.concatenate(temp_data_X, axis=0)
            majority_tr_data_Y = np.concatenate(temp_data_Y, axis=0)

        del temp_data_X, temp_data_Y

        temp_data_X = []
        temp_data_Y = []

        for j in minority_label:
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태
            a = np.arange(0,len(imbalance_dataset[0][j]))
            temp_set = np.delete(a, self.minor_validation_index)

            temp_data_X.append(imbalance_dataset[0][j][temp_set])
            temp_data_Y.append(imbalance_dataset[1][j][temp_set])

        minority_tr_data_X = np.concatenate(temp_data_X, axis=0)
        minority_tr_data_Y = np.concatenate(temp_data_Y, axis=0)

        if i == 10:
            imbalanced_data_X = minority_tr_data_X
            imbalanced_data_Y = minority_tr_data_Y
        else:
            imbalanced_data_X = np.concatenate((minority_tr_data_X, majority_tr_data_X), axis=0)
            imbalanced_data_Y = np.concatenate((minority_tr_data_Y, majority_tr_data_Y), axis=0)

        #save data for reconstruct the experiment result
        np.save("./checkpoint/imbalance_train_X", imbalanced_data_X)
        np.save("./checkpoint/imbalance_train_Y", imbalanced_data_Y)
        imbalance_trainset = [imbalanced_data_X, imbalanced_data_Y]

        del temp_data_X, temp_data_Y

        return imbalance_trainset

