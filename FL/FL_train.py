import os

import utils
from FL.FL_user import User
import copy
import torch
import FL.torch_dataset as datasets
import model
import numpy as np
import random
import torch.nn as nn

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


def train_model(config):
    # train_loss, train_acc = [], []

    list_val_loss_real, list_val_acc_real = [], []
    list_val_loss_syn, list_val_acc_syn = [], []
    server_logger = utils.setup_logger("server_logger", f"{config.folder_logger}/server.log", isServer=True)

    trainloader_list, valloader_list, testloader_real, testloader_synthetic = datasets.get_dataset(config)
    list_users = [User(trainloader=trainloader_list[idx],valloader=valloader_list[idx], index=idx, config=config) for idx in
                  range(config.total_num_users)]

    global_model = model.init_model(config)

    for round_ in range(config.num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round_, config.num_rounds - 1))

        if config.data_distribution != "iid" and round_ >= config.T_star:
            print("Freezing BN layers")
            freeze_stats(model)

        # train
        local_weights = []
        samples_per_client = []

        random_list = random.sample(range(config.total_num_users), config.users_per_round)

        for idx in random_list:
            user = list_users[idx]
            w, local_loss, local_correct, local_total = user.update_weights(
                model=copy.deepcopy(global_model), round_=round_)  # todo
            local_weights.append(copy.deepcopy(w))  # todo
            samples_per_client.append(local_total)

        global_weights = average_weights(local_weights, samples_per_client)
        global_model.load_state_dict(global_weights)

        # test on original, i.e., real, testset

        val_loss_real, val_accuracy_real = model_evaluation(config=config, model=global_model,
                                                            testloader=testloader_real, round_=round_,
                                                            ds="test_real")

        list_val_loss_real.append(val_loss_real)
        list_val_acc_real.append(val_accuracy_real)

        val_loss_syn, val_accuracy_syn = model_evaluation(config=config, model=global_model,
                                                          testloader=testloader_synthetic, round_=round_,
                                                          ds="test_synthetic")
        list_val_loss_syn.append(val_loss_syn)
        list_val_acc_syn.append(val_accuracy_syn)

        print('TEST REAL: Loss: {:.4f} Acc: {:.4f}'.format(val_loss_real, val_accuracy_real))
        print('TEST SYNTHETIC: Loss: {:.4f} Acc: {:.4f}'.format(val_loss_syn,val_accuracy_syn))

        server_logger.info(f"{round_},validation,{val_loss_real},{val_accuracy_real},{val_loss_syn},{val_accuracy_syn}")


def freeze_stats(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

def model_evaluation(config, model, testloader, round_, ds):
    labels_true = []
    labels_predicted = []
    labels_true_app = labels_true.extend
    labels_predicted_app = labels_predicted.extend
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        device = config.device
        running_loss = 0.0
        running_corrects = 0
        # running_total = 0

        for (i, data) in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            # running_total += labels.shape[0]
            labels_true_app(labels.cpu().numpy())
            labels_predicted_app(preds.cpu().numpy())


        epoch_loss = running_loss / len(testloader.dataset)
        epoch_acc = running_corrects / len(testloader.dataset)

        np.savez_compressed(f"{config.folder_logger}/class_statistics/{ds}/server_round{round_}.npz",
                            true=np.array(labels_true), predicted=np.array(labels_predicted))

        return epoch_loss, epoch_acc


def average_weights(w, samples_per_client):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = torch.true_divide(w[i][key], 1 / samples_per_client[i])
            else:
                w_avg[key] += torch.true_divide(w[i][key], 1 / samples_per_client[i])
        w_avg[key] = torch.true_divide(w_avg[key], sum(samples_per_client))
    return w_avg
