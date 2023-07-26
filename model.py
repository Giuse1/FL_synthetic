from torch import nn
import torch.nn.functional as F
import torchvision
import torch
import copy
from functools import reduce

# class cifar(nn.Module):
#     def __init__(self, num_channels, num_classes):
#         super(cifar, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
#         self.fc1 = nn.Linear(1024, 64)
#         self.fc2 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # print(x.shape)
#
#         x = F.relu(F.max_pool2d(x, 2))
#         # print(x.shape)
#         x = self.conv2(x)
#         # print(x.shape)
#         x = F.relu(F.max_pool2d(x, 2))
#         # print(x.shape)
#         x = self.conv3(x)
#         # print(x.shape)
#         x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
#         # print(x.shape)
#         x = F.relu(self.fc1(x))
#         # print(x.shape)
#         x = self.fc2(x)
#         return x


class LeNet5(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        in_channels = 1

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


import torch
import torch.nn as nn
import numpy as np



def init_model(config):

    if config.model == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, out_features=10, bias=True)

    elif config.model == "resnet18_gn":
        model = get_resnet18_gn()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, out_features=10, bias=True)


    elif config.model == "vgg11":
        model = torchvision.models.vgg11(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, out_features=10, bias=True)


    elif config.model == "lenet":
        model = LeNet5(num_classes=10)

    else:
        raise Exception("No correct model has been selected")

    print(model)
    return model.to(config.device)


def get_resnet18_gn():
        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)

        net = torchvision.models.resnet18(pretrained=False)

        cp_net = copy.deepcopy(net)
        for name, module in cp_net.named_modules():
            if isinstance(module, nn.BatchNorm2d):

                print(name)
                bn = get_module_by_name(net, name)

                gn = nn.GroupNorm(1, bn.num_features)

                if len(name.split(".")) == 1:
                    net._modules[name] = gn
                else:

                    try:
                        layer, idx, last = name.split(".")

                        setattr(net._modules[layer][int(idx)], last, gn)
                    except:
                        layer, idx, last, lastidx = name.split(".")
                        setattr(net._modules[layer][int(idx)]._modules[last], lastidx, gn)



        return net
