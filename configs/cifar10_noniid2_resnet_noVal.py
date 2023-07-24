from configs.cifar10_default import get_default_config, plot_config, get_config_folder


def get_config():
    config = get_default_config()

    config.data_distribution = "non_iid_2"
    config.model = "resnet18"

    config.validation_node = False
    config.T_star = 3

    config.folder_logger = get_config_folder(config)

    plot_config(config)

    return config
