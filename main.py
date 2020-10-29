import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
import os

from trainer import *
from data.data_cifar10 import *
from utils import *

dir10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"

cifar10 = Cifar10(directory=dir10)
train_X, train_Y, test_X, test_Y = cifar10.initialize(directory=dir10)
test_X = cifar10.ordered_test_X
test_Y = cifar10.ordered_test_Y

balanced_trainset = [train_X, train_Y]
balanced_testset = [test_X, test_Y]

#custom_index = np.load('./checkpoint/data_idx/bal_L2myAvg_idx_99.npy')
#soft_balanced_hard_idx = np.load('./checkpoint/sb_hard_idx75.npy')

#forgettable_X = train_X[custom_index]
#forgettable_Y = train_Y[custom_index]

#forgettable_data = [forgettable_X, forgettable_Y]

trainer_resnet = resnet_TwoPhase(trainset=balanced_trainset, testset=balanced_testset, lr=0.1, lr_decay=[60,80], imbalance_ratio="L1EasyWAvgRm99")
trainer_resnet.first_phase_run(epochs = 100)
