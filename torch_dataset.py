import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torchvision.transforms as transforms
from PIL import Image

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

NUM_WORKERS = 1
PIN_MEMORY = True


class FolderDataset2(Dataset):

    def __init__(self, config, folder, transform):

        self.folder = folder
        if folder == "synthetic":
            self.folder = os.path.join(self.folder, f"{config.corrector}/cfg{str(config.cfg_scale)}")

        subfolder = os.path.join(f"{config.root}/dataset_raw/{config.dataset}", self.folder)

        self.list_files = os.listdir(subfolder)

        self.list_files = [os.path.join(subfolder, x) for x in self.list_files]
        self.data = []
        self.targets = []

        if folder == "synthetic":
            self.get_synthetic()
        else:
            self.get_original()

        self.data = np.stack(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        image, label = self.data[idx], self.targets[idx]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        # sample = {0: image, 1: label}

        return image, label

    def get_original(self):

        for i, f in enumerate(self.list_files):
            to_concat = np.load(f)
            # data_to_concat = np.transpose(to_concat['samples'], (0, 3, 1, 2))
            data_to_concat = to_concat['samples']
            # data_to_concat = np.divide(data_to_concat, 255)

            lebels_to_concat = to_concat['labels']

            self.data.extend(data_to_concat)
            self.targets.extend(lebels_to_concat)

    def get_synthetic(self):

        for i, f in enumerate(self.list_files):

            splitted = f.split('_')
            label = int(splitted[-2])

            idx = int(splitted[-1].split('.')[0])

            to_concat = np.load(f)['samples']
            if idx == 7:
                to_concat = to_concat[:1000 - 128 * 7]
            # to_concat = np.divide(to_concat, 255)
            # to_concat = np.transpose(to_concat, (0, 3, 1, 2))

            self.targets.extend([label] * to_concat.shape[0])
            self.data.extend(to_concat)


def get_dataset(config):
    if config.dataset == "cifar10":
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    elif config.dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
    else:
        raise Exception("Wrong dataset")

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    trainset = FolderDataset2(config, "train", train_transform)
    testset_real = FolderDataset2(config, "test", test_transform)
    testset_synthetic = FolderDataset2(config, "synthetic", test_transform)

    testloader_real = torch.utils.data.DataLoader(testset_real, batch_size=config.batch_size, shuffle=False,
                                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    testloader_synthetic = torch.utils.data.DataLoader(testset_synthetic, batch_size=config.batch_size, shuffle=False,
                                                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if config.data_distribution == "iid":
        trainloader_list, valloader_list = get_train_iid(config, trainset)
    elif config.data_distribution == "non_iid_1":
        trainloader_list, valloader_list = get_train_one_class_per_user(config, trainset)
    elif config.data_distribution == "non_iid_2":
        trainloader_list, valloader_list = get_train_two_classes_per_user(config, trainset)

    else:
        raise Exception("Wrong data distribution")

    return trainloader_list, valloader_list, testloader_real, testloader_synthetic


def get_subset(set, indexes, config):

    if config.validation_node:
        random.shuffle(indexes)
        len_train = int(0.85*len(indexes))
        train_indexes = indexes[:len_train]
        val_indexes = indexes[len_train:]
        trainset = Subset(set, train_indexes)
        valset = Subset(set, val_indexes)
    else:
        trainset = Subset(set, indexes)
        valset = None

    return trainset, valset


def get_train_iid(config, trainset):
    total_data = len(trainset)

    random_list = random.sample(range(total_data), total_data)
    data_per_client = int(total_data / config.total_num_users)
    trainloader_list = []
    valloader_list = [None]*config.total_num_users

    for i in range(config.total_num_users):
        indexes = random_list[i * data_per_client: (i + 1) * data_per_client]
        train_ds, val_ds = get_subset(trainset, indexes, config)
        trainloader_list.append(
            torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY))
        if val_ds:
            valloader_list[i] = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY)
    return trainloader_list, valloader_list


def get_train_one_class_per_user(config, trainset):
    n_classes = 10

    targets = np.array(trainset.targets)

    target_indices = np.arange(len(targets))
    trainloader_list = []
    valloader_list = [None]*config.total_num_users

    for c in range(n_classes):
        idx_class = targets == c
        train_idx = target_indices[idx_class]

        # here code to split dataset if you want more than 10 clients

        train_ds, val_ds = get_subset(trainset, train_idx, config)

        trainloader_list.append(
            torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY))

        if val_ds:
            valloader_list[c] = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY)

    return trainloader_list, valloader_list


def get_train_two_classes_per_user(config, trainset):
    samples_per_class = int(len(trainset) / config.total_num_users / 2)

    list_partitions = []
    for i in range(config.total_num_users):
        list_partitions.append((i, 0))
        list_partitions.append((i, samples_per_class))

    targets = np.array(trainset.targets)

    target_indices = np.arange(len(targets))
    trainloader_list = []
    valloader_list = [None]*config.total_num_users

    len_dict = {}
    for i in range(config.total_num_users):

        train_idx_2_classes = []

        for _ in range(2):
            rnd_idx = random.randint(0, len(list_partitions) - 1)
            c, starting_idx_samples = list_partitions[rnd_idx]
            del list_partitions[rnd_idx]

            idx_class = targets == c
            train_idx = target_indices[idx_class]
            if c not in len_dict:
                len_dict[c] = int(len(train_idx) / 2)
                starting_idx_samples = 0
                ending_idx_samples = len_dict[c]
            else:
                starting_idx_samples = len_dict[c]
                ending_idx_samples = len(train_idx)
            print((starting_idx_samples, ending_idx_samples))
            train_idx_2_classes.append(train_idx[starting_idx_samples:ending_idx_samples])

        # here code to split dataset if you want more than 10 clients
        train_idx_2_classes = np.concatenate(train_idx_2_classes)
        train_ds, val_ds = get_subset(trainset, train_idx_2_classes, config)

        trainloader_list.append(
            torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY))

        if val_ds:
            valloader_list[i]=torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY)


    return trainloader_list, valloader_list

# class FolderDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, folder, transform=None):
#         """
#         Arguments:
#         img_{class}_idx.png
#         """
#         self.folder = folder
#         self.list_images = sorted(os.listdir(folder))
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.list_images)
#
#     def __getitem__(self, idx):
#         img_name = self.list_images[idx]
#         image = np.load(os.path.join(self.folder, img_name))
#         label = int(img_name.split('_')[1])
#         if self.transform:
#             image = self.transform(image)
#
#         sample = {'image': image, 'targets': label}
#
#         return sample
#

# def load_dataset(config):
#     """
#     load cifar dataset
#     :return:
#     """
#     if config.dataset == "cifar10":
#         mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
#         folder = "/home/giuse/Desktop/generated_cifar_last/26/langevin/cfg_2/"
#     else:
#         mean, std = (0.1307,), (0.3081,)
#         folder = "/home/giuse/Desktop/generated_mnist_last/26/langevin/cfg_2/"
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize(mean, std)])
#
#     trainset = torchvision.datasets.CIFAR10(root='/home/giuse/datasets', train=True, download=False,
#                                             transform=transform)
#
#
#     testset_real = torchvision.datasets.CIFAR10(root='/home/giuse/datasets', train=False, download=False,
#                                            transform=transform)
#
#     testset_synthetic = torchvision.datasets.CIFAR10(root='/home/giuse/datasets', train=False, download=False,
#                                                 transform=transform)
#
#     # trainset = torchvision.datasets.ImageFolder(root=folder,
#     #                                             transform=transform)  # todo
#     #
#     # testset_real = torchvision.datasets.ImageFolder(folder, transform=transform)  # todo
#     # testset_synthetic = torchvision.datasets.ImageFolder(folder, transform=transform)  # todo
#
#     return trainset, testset_real, testset_synthetic
