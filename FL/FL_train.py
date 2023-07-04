from FL.FL_user import User
import copy
import torch
from FL.torch_dataset import get_cifar_iid, cifar_one_class_per_user
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



def train_model(global_model, criterion, num_rounds, local_epochs, total_num_users, num_users, batch_size,
                learning_rate, decay, iid):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    if iid:
        trainloader_list, valloader = get_cifar_iid(batch_size=batch_size, total_num_clients=total_num_users)
    else:
        trainloader_list, valloader = cifar_one_class_per_user(batch_size=batch_size, total_num_clients=total_num_users,
                                                               shuffle=True)

    for round_ in range(num_rounds):
        print('-' * 10)
        print('Epoch {}/{}'.format(round_, num_rounds - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                local_weights = []
                samples_per_client = []

                random_list = random.sample(range(total_num_users), num_users)

                for idx in random_list:
                    local_model = User(dataloader=trainloader_list[idx], index=idx, criterion=criterion,
                                       local_epochs=local_epochs, learning_rate=learning_rate, decay=decay)
                    w, local_loss, local_correct, local_total = local_model.update_weights(
                        model=copy.deepcopy(global_model).double(), epoch=round_)
                    local_weights.append(copy.deepcopy(w))
                    samples_per_client.append(local_total)

                global_weights = average_weights(local_weights, samples_per_client)
                global_model.load_state_dict(global_weights)

            else:
                val_loss_r, val_accuracy_r = model_evaluation(model=global_model.double(),
                                                              dataloader=valloader, criterion=criterion)

                val_loss.append(val_loss_r)
                val_acc.append(val_accuracy_r)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss_r, val_accuracy_r))

    return train_loss, train_acc, val_loss, val_acc


def model_evaluation(model, dataloader, criterion):
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for (i, data) in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs.double())
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            running_total += labels.shape[0]

        epoch_loss = running_loss / running_total
        epoch_acc = running_corrects.double() / running_total

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
