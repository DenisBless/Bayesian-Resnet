import torch.nn as nn
from bayesian_layer_resnet import FlattenLayer, Bayesian_conv2D, Bayesian_fullyconnected

class Small_conv_net(nn.Module):
    def __init__(self, outputs, inputs):
        super(Small_conv_net, self).__init__()

        self.conv1 = Bayesian_conv2D(inputs, 6, 5, stride=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Bayesian_conv2D(6, 16, 5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = Bayesian_fullyconnected(5 * 5 * 16, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.soft3 = nn.Softplus()

        self.fc2 = Bayesian_fullyconnected(120, outputs)


    def forward(self, x):
        'Forward pass with Bayesian weights'
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.soft1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.soft2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.soft3(x)
        x = self.fc2(x)
        return x


