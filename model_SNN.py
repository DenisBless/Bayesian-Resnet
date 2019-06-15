import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from bayesian_layer_resnet import FlattenLayer
from torchsummary import summary



# batch_size = 16
# transform = transforms.Compose([
#     transforms.ToTensor()])
# train_data = datasets.MNIST('data', train=True,
#                               download=True, transform=transform)
# test_data = datasets.MNIST('data', train=False,
#                              download=True, transform=transform)
#
# # prepare data loaders (combine dataset and sampler)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
#
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)

        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.soft3 = nn.Softplus()
        self.fc2 = nn.Linear(120, 10)



    def forward(self, x):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

# summary(model, (3, 32, 32))