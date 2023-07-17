import ml_collections
import torch
import os

from configs.cifar10_default import get_default_config, plot_config


def get_config():
    config = get_default_config()

    config.data_distribution = "non_iid_1"
    config.model = "resnet18"

    config.folder_logger = f"{config.root}/reports/{config.dataset}/val_node_{config.validation_node}/" \
                           f"{config.data_distribution}_{config.model}_{config.optimizer}_{config.learning_rate}/" \
                           f"cfg{config.cfg_scale}"

    plot_config(config)

    return config
