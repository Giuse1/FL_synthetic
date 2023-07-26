import torch
import numpy as np
import random
import torch.nn as nn
import utils
from operator import itemgetter

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class User(object):
    def __init__(self, trainloader, valloader, index, config):

        self.trainloader = trainloader
        self.valloader = valloader

        self.classes = list(set(list(itemgetter(*self.trainloader.dataset.indices)(self.trainloader.dataset.dataset.targets))))
        if len(self.classes) != 10:
            print("USING MAKSED CROSS-ENTROPY LOSS")
        self.id = index
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate
        self.decay = config.decay

        self.logger = utils.setup_logger(f"client{index}", f"{config.folder_logger}/client{index}.log", isServer=False)
        self.config = config

    def update_weights(self, model, round_):

        if self.valloader is not None:
            self.test_model(model, round_, "before_training")

        model.train()
        lr = self.learning_rate * self.decay ** round_
        optimizer = eval(f"torch.optim.{self.config.optimizer}(model.parameters(), lr=lr)")

        labels_true = []
        labels_predicted = []
        labels_true_app = labels_true.extend
        labels_predicted_app = labels_predicted.extend

        for _ in range(self.local_epochs):

            local_correct = 0
            local_loss = 0.0
            for (i, data) in enumerate(self.trainloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)

                if len(self.classes) != 10:
                    label_mask = torch.zeros(10, device=self.config.device)
                    label_mask[self.classes] = 1
                    log_probs = log_probs.masked_fill(label_mask == 0, 0)

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

        self.logger.info(f"{round_},train,{local_loss},{local_correct},{len(self.trainloader.dataset)}")

        if self.valloader is not None:
            self.test_model(model, round_, "after_training")

        return model.state_dict(), local_loss, local_correct, len(self.trainloader.dataset)

    def test_model(self, model, round_, step):

        labels_true = []
        labels_predicted = []
        labels_true_app = labels_true.extend
        labels_predicted_app = labels_predicted.extend

        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_corrects = 0
            # test_total = 0

            for (i, data) in enumerate(self.valloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)

                if len(self.classes) != 10:
                    label_mask = torch.zeros(10, device=self.config.device)
                    label_mask[self.classes] = 1
                    outputs = outputs.masked_fill(label_mask == 0, 0)

                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels)
                # test_total += labels.shape[0]
                labels_true_app(labels.cpu().numpy())
                labels_predicted_app(preds.cpu().numpy())

            epoch_loss = test_loss # / len(self.valloader.dataset)
            epoch_acc = test_corrects # / len(self.valloader.dataset)

            self.logger.info(f"{round_},validation_{step},{epoch_loss},{epoch_acc},{len(self.valloader.dataset)}")
            np.savez_compressed(
                f"{self.config.folder_logger}/class_statistics/validation_{step}/client{self.id}_round{round_}.npz",
                true=np.array(labels_true), predicted=np.array(labels_predicted))

            return epoch_loss, epoch_acc



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

def freeze_stats_(model):

    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            # if hasattr(module, 'weight'):
                module.track_running_stats = False
                # print(module)
                # module.weight.requires_grad_(False)
            # if hasattr(module, 'bias'):
            #     # module.bias.requires_grad_(False)
            #     print(module)

    return model
