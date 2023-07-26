import ml_collections
import torch
import os


def get_default_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.root = "/home/gdigiacomo/FL_diffusion"  # "/home/giuse/Desktop/FL"  # "FL_diffusion"  # todo

    config.stats_before = False

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config.total_num_users = 10
    config.users_per_round = 10

    config.num_rounds = 150
    config.local_epochs = 1

    config.optimizer = "Adam"
    config.learning_rate = 0.0001
    config.decay = 1  # to consider

    config.batch_size = 64  # todo

    config.dataset = "cifar10"

    config.cfg_scale = 1
    config.corrector = "langevin"  # to consider

    return config


def get_config_folder(config):
    if config.stats_before is True:
        stats_before = "stats_before_"
    else:
        stats_before = ""

    if "T_star" in config:
        tstar_str = f"Tstar_{config.T_star}_"
    else:
        tstar_str = ""

    if "algorithm" in config:
        alg_str = f"{config.algorithm}_"
    else:
        alg_str = ""

    if "masked" in config:
        masked_str = f"maskedCE_"
    else:
        masked_str = ""

    s = f"{config.root}/reports/{config.dataset}/val_node_{config.validation_node}/{stats_before}" \
        f"{config.data_distribution}_{alg_str}{masked_str}{tstar_str}{config.model}_{config.optimizer}_{config.learning_rate}/" \
        f"cfg{config.cfg_scale}"

    return s


def plot_config(config):
    os.makedirs(f"{config.folder_logger}/class_statistics/train", exist_ok=False)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_real", exist_ok=False)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_synthetic", exist_ok=False)

    os.makedirs(f"{config.folder_logger}/class_statistics/validation_before_training", exist_ok=False)
    os.makedirs(f"{config.folder_logger}/class_statistics/validation_after_training", exist_ok=False)

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
    print(f"node validation: {config.validation_node}")
