import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from data.data_cifar10 import *
from data.data_preprocessing import *
import model.resnet as model_res

from utils import *
import os
import copy
from tqdm import tqdm_notebook
import math
import torch.nn.functional as F

class resnet_TwoPhase:
    def __init__(self, trainset=None, testset=None, lr=None, lr_decay = None, batch_norm=None, imbalance_ratio=None, mu=None):
        self.trainset = trainset
        self.testset = testset
        self.lr = lr
        self.lr_decay = lr_decay
        # self.epoch = epoch
        self.imbalance_ratio = imbalance_ratio

        self.optimzier = None
        self.scheduler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = batch_norm
        if batch_norm == True:
            self.net = model_res.resnet32().cuda()
        else:
            self.net = model_res.resnet32().cuda()

        self.transform_train = transforms.Compose([transforms.ToTensor(),
                                                   transforms.ToPILImage(),
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5071, 0.4866, 0.4409],
                                                                        [0.2673, 0.2564, 0.2762])
                                                   ])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.5071, 0.4866, 0.4409],
                                                                       [0.2673, 0.2564, 0.2762])
                                                  ])

        train_dataset = CustomDataset(self.trainset[0], self.trainset[1], transform=self.transform_train)
        test_dataset = CustomDataset(self.testset[0], self.testset[1], transform=self.transform_test)

        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        self.training_class_acc = []
        self.test_class_acc = []

        self.epoch_val_acc_list = []
        self.epoch_val_loss_list = []
        self.epoch_class_acc = []

        self.prev_forgetting = np.zeros(trainset[0].__len__())
        self.new_forgetting = np.zeros(trainset[0].__len__())
        self.count_forgetting = np.zeros(trainset[0].__len__())

        self.cumulative_training_acc = torch.tensor([]).cuda()
        self.cumulative_training_target = torch.tensor([]).cuda()

        self.grad_list = []
        self.temp_input_grad = np.zeros(trainset[0].__len__())

        self.prob_list = []
        self.training_prob = np.zeros(trainset[0].__len__())

    def first_phase_train(self, trainloader=None, current_epoch=None):
        '''

        :param model: model trained with weighted sampling data
        :param trainloader: weighted sampling dataset
        :param current_epoch:
        :return:
        '''

        print('\nEpoch: %d' % current_epoch)
        print("Training...")
        #model = model

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        self.training_target = torch.tensor([]).cuda()
        self.training_output = torch.tensor([]).cuda()

        temp_input_grad = self.temp_input_grad.copy()
        temp_prob = self.training_prob.copy()

        for batch_idx, (inputs, targets, traindata_idx) in enumerate(trainloader):

            inputs = inputs.float().cuda()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            ce_loss = criterion(outputs, targets)
            loss = ce_loss

            loss.backward()
            # print(inputs.grad)

            input_grad = inputs.grad
            input_grad = input_grad.view(traindata_idx.shape[0], -1)
            input_grad = torch.norm(input_grad, dim=1, p=1)

            # print(input_grad.shape, traindata_idx.shape)
            temp_input_grad[traindata_idx] = input_grad.cpu()

            prob = F.softmax(outputs, dim=1).cpu()

            temp_shape = np.arange(outputs.__len__())

            temp_prob[traindata_idx] = prob[temp_shape, targets].detach()
            # print(prob[:,targets].shape)

            self.optimizer.step()

            _, predicted = outputs.max(1)

            self.training_output = torch.cat(
                (self.training_output.type(dtype=torch.long), predicted)
                , dim=0)

            self.training_target = torch.cat(
                (self.training_target.type(dtype=torch.long), targets)
                , dim=0)

            correct_idx = np.array(torch.where(predicted.eq(targets) == True)[0].cpu())
            wrong_idx = np.array(torch.where(predicted.eq(targets) == False)[0].cpu())

            correct_idx = traindata_idx[correct_idx]
            wrong_idx = traindata_idx[wrong_idx]

            # ***count forgetting rate**** very important
            if current_epoch == 1:
                self.prev_forgetting[correct_idx] = 1

            elif current_epoch > 1:
                self.new_forgetting[correct_idx] = 1
                self.new_forgetting[wrong_idx] = 0

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            '''
            if current_epoch % 5 == 4:
                if batch_idx % 10 == 9:  # print every 2000 mini-batches
                    print('Training | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                          (current_epoch + 1, batch_idx + 1, train_loss / (batch_idx + 1), correct / total))
            '''

        if current_epoch > 1 :
            for j in range(len(self.prev_forgetting)):
                if self.prev_forgetting[j] == 1 and self.new_forgetting[j] == 0:
                    self.count_forgetting[j] = self.count_forgetting[j] + 1


            self.prev_forgetting = self.new_forgetting.copy()

        self.cumulative_training_acc = torch.cat((self.cumulative_training_acc.type(dtype=torch.long),
                                                 self.training_output), dim=0)
        self.cumulative_training_target = torch.cat((self.cumulative_training_target.type(dtype=torch.long),
                                                    self.training_target), dim=0)

        if current_epoch < 100:
            self.grad_list.append(temp_input_grad)



    def one_epoch_test(self, testloader=None, current_epoch=None):
        global best_acc
        criterion = nn.CrossEntropyLoss()

        print('\nEpoch: %d' % current_epoch)
        print("Testing...")

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, data_idx) in enumerate(testloader):
                print(self.scheduler.get_lr())
                inputs = inputs.float().cuda()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                class_correct = predicted.eq(targets).sum().item()
                correct += predicted.eq(targets).sum().item()
                # print(total)
                # print(outputs.shape)
                print('Test | [%d, %5d] loss: %.3f total accuracy : %.3f' %
                      (current_epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), correct / total))

        self.epoch_test_loss = test_loss / (batch_idx + 1)
        self.epoch_test_correct = correct / total

        self.epoch_val_acc_list.append(self.epoch_test_correct)
        self.epoch_val_loss_list.append(self.epoch_test_loss)

        # Save checkpoint.
        acc = 100. * correct / total

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': current_epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.batch_norm == True:
                torch.save(state,
                           './checkpoint/ckpt_res32_bn_{}.pth'.format(self.imbalance_ratio))
            else :
                torch.save(state, './checkpoint/ckpt_res32_{}.pth'.format(self.imbalance_ratio))

            best_acc = acc

    def first_phase_run(self, epochs):
        global best_acc

        best_acc = 0
        #set_seed(2020)


        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9,
                                   weight_decay=5e-4)

        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.lr_decay, gamma=0.1)

        for epoch in range(epochs):
            self.first_phase_train(self.trainloader, epoch)
            self.one_epoch_test(self.testloader, epoch)
            self.scheduler.step()

###################################################################################################################################################
