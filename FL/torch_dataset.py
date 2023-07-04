import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import random


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def get_cifar():
    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root='/home/giuse/datasets', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/home/giuse/datasets', train=False, download=True, transform=transform)

    return trainset, testset

def get_cifar_iid(batch_size, total_num_clients=10):

    trainset, testset = get_cifar()

    total_data = len(trainset)

    random_list = random.sample(range(total_data), total_data)
    data_per_client = int(total_data / total_num_clients)
    trainloader_list = []

    for i in range(total_num_clients):

        indexes = random_list[i*data_per_client: (i+1)*data_per_client]
        d = Subset(trainset, indexes)
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader_list, testloader


def cifar_one_class_per_user(batch_size):
    n_classes = 10
    trainset, testset = get_cifar()


    #all_set = torch.utils.data.ConcatDataset([trainset, testset])
    l = int(len(trainset)/2)
    fl_set, _ = torch.utils.data.random_split(trainset, [l, l],generator=torch.Generator().manual_seed(42))
    print(len(fl_set))


    targets = np.array([sample[1] for sample in fl_set])
    target_indices = np.arange(len(targets))
    trainloader_list = []
    for c in range(n_classes):
        idx_class = targets == c
        train_idx = target_indices[idx_class]

        # here code to split dataset if you want more than 10 clients

        d = Subset(fl_set, train_idx)
        trainloader_list.append(torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True))

        print(len(d))

    return trainloader_list

