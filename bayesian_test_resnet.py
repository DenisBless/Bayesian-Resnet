from __future__ import print_function

import math
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
from utils_BNN_resnet import neg_ELBO, Logger
from model_BNN_test import Small_conv_net
from BayesianResnet import resnet18
import torchvision.models as models

use_cuda = torch.cuda.is_available()

learning_rate = 0.001
weight_decay = 0
batch_size = 16
num_epochs = 5


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# # choose the training and test datasets
# train_data = datasets.MNIST('data', train=True,
#                               download=True, transform=transform)
# test_data = datasets.MNIST('data', train=False,
#                              download=True, transform=transform)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
net = Small_conv_net(10, 3)
#net = resnet18(pretrained=False)
# net = models.resnet18(pretrained=False)

if use_cuda:
    net.cuda()

neg_elbo = neg_ELBO(net=net)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


for epoch in range(1, num_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    m_train = math.ceil(len(train_data) / batch_size)
    m_test = math.ceil(len(test_data) / batch_size)

    net.train()

    total = 0
    correct = 0

    for batch_idx, (data, target) in zip(tqdm(range(m_train)), (train_loader)):
        # move tensors to GPU if CUDA is available
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        beta = 1 / batch_size

        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # calculate the batch loss
        loss = neg_elbo(output, target, beta)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += (loss.item() * data.size(0))

    train_loss = train_loss / len(train_loader.dataset)



    print('--------------------------------------------------------------')
    print('Epoch:', epoch)
    print('--------------------------------------------------------------')
    print('Trainig loss:', train_loss)
    print('--------------------------------------------------------------')
    print('Accuracy of the network on the train images: {} percent ({}/{})'.format(
        100 * correct / total, correct, total))
    print('--------------------------------------------------------------')
