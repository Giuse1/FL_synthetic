import ml_collections
import torch
import os


from configs.cifar10_default import get_default_config


def get_config():
    config = get_default_config()

    config.seed = 0
    config.root = "/home/giuse/Desktop/FL"  # "/home/giuse/Desktop/FL"  # "FL_diffusion"  # todo

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config.total_num_users = 10
    config.users_per_round = 10

    config.num_rounds = 100
    config.local_epochs = 1

    config.optimizer = "Adam"
    config.learning_rate = 0.0001
    config.decay = 1  # to consider

    config.batch_size = 64 # todo

    config.data_distribution = "iid"
    config.dataset = "cifar10"

    config.model = "vgg11"

    config.cfg_scale = 1
    config.corrector = "langevin"  # to consider

    config.validation_node = False

    config.folder_logger = f"{config.root}/reports/{config.dataset}/val_node_{config.validation_node}/" \
                           f"{config.data_distribution}_{config.model}_{config.optimizer}_{config.learning_rate}/" \
                           f"cfg{config.cfg_scale}"

    os.makedirs(f"{config.folder_logger}/class_statistics/train", exist_ok=True)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_real", exist_ok=True)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_synthetic", exist_ok=True)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_synthetic", exist_ok=True)

    if config.validation_node:
        os.makedirs(f"{config.folder_logger}/class_statistics/validation_before_training", exist_ok=True)
        os.makedirs(f"{config.folder_logger}/class_statistics/validation_after_training", exist_ok=True)


    print(f"total_num_users: {config.total_num_users}")
    print(f"users_per_round: {config.users_per_round}")
    print(f"num_rounds: {config.num_rounds}")
    print(f"local_epochs: {config.local_epochs}")
    print(f"optimizer: {config.optimizer}")
    print(f"batch_size: {config.batch_size}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"decay: {config.decay}")
    print(f"data_distribution: {config.data_distribution}")
    print(f"model: {config.model}")

    return config
