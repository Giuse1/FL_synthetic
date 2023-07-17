import os

from configs.cifar10_default import get_default_config


def get_config():
    config = get_default_config()

    config.data_distribution = "non_iid_2"
    config.model = "resnet18"

    config.folder_logger = f"{config.root}/reports/{config.dataset}/val_node_{config.validation_node}/" \
                           f"{config.data_distribution}_{config.model}_{config.optimizer}_{config.learning_rate}/" \
                           f"cfg{config.cfg_scale}"

    os.makedirs(f"{config.folder_logger}/class_statistics/train", exist_ok=True)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_real", exist_ok=True)
    os.makedirs(f"{config.folder_logger}/class_statistics/test_synthetic", exist_ok=True)

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
