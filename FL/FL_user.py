import os

import torch
import numpy as np
import random
import torch.nn as nn
import utils

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class User(object):
    def __init__(self, dataloader, index, config):

        self.dataloader = dataloader
        self.id = index
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate
        self.decay = config.decay



        self.logger = utils.setup_logger(f"client{index}", f"{config.folder_logger}/client{index}.log", isServer=False)
        self.config = config


    def update_weights(self, model, round_):
        model.train()
        lr = self.learning_rate * self.decay**round_
        optimizer = eval(f"torch.optim.{self.config.optimizer}(model.parameters(), lr=lr)")

        labels_true = []
        labels_predicted = []
        labels_true_app = labels_true.extend
        labels_predicted_app = labels_predicted.extend

        for _ in range(self.local_epochs):

            local_correct = 0
            local_loss = 0.0
            for (i, data) in enumerate(self.dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                _, preds = torch.max(log_probs, 1)
                local_correct += torch.sum(preds == labels).cpu().numpy()

                loss.backward()
                optimizer.step()
                local_loss += loss.item() * images.size(0)

                labels_true_app(labels.cpu().numpy())
                labels_predicted_app(preds.cpu().numpy())


        np.savez_compressed(f"{self.config.folder_logger}/class_statistics/train/client{self.id}_round{round_}.npz",
                            true=np.array(labels_true), predicted=np.array(labels_predicted))


        self.logger.info(f"{round_},train,{local_loss},{local_correct},{len(self.dataloader.dataset)}")

        return model.state_dict(), local_loss, local_correct, len(self.dataloader.dataset)

    def test_model(self, model, criterion):
        with torch.no_grad():

            model.eval()
            test_loss = 0.0
            test_corrects = 0
            # test_total = 0

            for (i, data) in enumerate(self.testloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels)
                # test_total += labels.shape[0]

            epoch_loss = test_loss / len(self.testloader.dataset)
            epoch_acc = test_corrects / len(self.testloader.dataset)

            return epoch_loss, epoch_acc


