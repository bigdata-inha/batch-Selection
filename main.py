import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
import os

from trainer import *
from data.dataset_cifar10 import *
from utils import *

dir10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"