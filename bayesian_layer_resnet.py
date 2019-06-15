import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
from utils_BNN_resnet import kl_gauss_prior

cuda = torch.cuda.is_available()

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class Abs(ModuleWrapper):

    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return x.abs_()


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class Bayesian_conv2D(ModuleWrapper):
    """
    Module for bayesian inference in a convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False):  # TODO add bias, prior
        super(Bayesian_conv2D, self).__init__()
        # Params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Init weights
        self.mean = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.logvar = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        # Bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv2d_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding,
                                                          self.dilation,
                                                          self.groups)
        self.conv2d_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding,
                                                            self.dilation,
                                                            self.groups)

        # Initialize all parameters
        self.reset_params()

    def reset_params(self):
        """
        Fills the placeholder variables with values.
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.stdv = stdv
        self.mean.data.uniform_(-stdv, stdv)
        self.logvar.data.uniform_(0.5, 1.5)
        self.logvar.data.log_()

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        # Data term
        # Two consecutive convolutions for mean and std of the activations (Local reparameterization trick)
        lrt_mean = self.conv2d_bias(x, self.mean)
        lrt_std = torch.sqrt(1e-16 + self.conv2d_nobias(x * x, torch.exp(self.logvar)))
        eps = torch.cuda.FloatTensor(lrt_std.size()).normal_()
        return lrt_mean + eps * lrt_std

    def kl_div(self):
        # D_KL(q||p) where p ~ N(0,1)
        q_std = torch.sqrt(torch.exp(self.logvar))
        return kl_gauss_prior(q_mean=self.mean,
                              q_std=q_std,
                              p_mean=0,
                              p_std=np.sqrt(self.stdv))

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

#
class Bayesian_fullyconnected(ModuleWrapper):
    # TODO: add bias
    def __init__(self, in_features, out_features, bias=False):
        super(Bayesian_fullyconnected, self).__init__()

        # Params
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        # Specify weight and prior distribution
        self.mean = Parameter(torch.Tensor(out_features, in_features))
        self.logvar = Parameter(torch.Tensor(out_features, in_features))

        # Initialize all parameters
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: unkown rule of thumb, look in pytorch implem. maybe kaiming init?
        stdv = 1.0 / math.sqrt(self.logvar.size(1))
        self.stdv=stdv
        self.mean.data.uniform_(-stdv, stdv)
        self.logvar.data.uniform_(0.5, 1.5)
        self.logvar.data.log_()

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        # Data term
        lrt_mean = F.linear(x, self.mean)
        if self.bias is not None:
            lrt_mean = lrt_mean + self.bias
        lrt_std = torch.sqrt(1e-16 + F.linear(x * x, torch.exp(self.logvar)))
        eps = torch.cuda.FloatTensor(lrt_std.size()).normal_()
        return lrt_mean + eps * lrt_std

    def kl_div(self):
        # D_KL(q||p) where p ~ N(0,1)
        q_std = torch.sqrt(torch.exp(self.logvar))
        return kl_gauss_prior(q_mean=self.mean,
                              q_std=q_std,
                              p_mean=0,
                              p_std=np.sqrt(self.stdv))
