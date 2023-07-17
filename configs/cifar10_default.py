import ml_collections
import torch
import os


def get_default_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.root = "/home/giuse/Desktop/FL"  # "/home/giuse/Desktop/FL"  # "FL_diffusion"  # todo

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config.total_num_users = 10
    config.users_per_round = 10

    config.num_rounds = 100
    config.local_epochs = 1

    config.optimizer = "SGD"
    config.learning_rate = 0.01
    config.decay = 1  # to consider

    config.batch_size = 64 # todo

    config.dataset = "cifar10"


    config.cfg_scale = 1
    config.corrector = "langevin"  # to consider

    config.validation_node = False


    return config
