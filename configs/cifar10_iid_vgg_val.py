from configs.cifar10_default import get_default_config, plot_config, get_config_folder


def get_config():
    config = get_default_config()

    config.data_distribution = "iid"
    config.model = "vgg11"
    config.validation_node = True

    config.folder_logger = get_config_folder(config)
    plot_config(config)

    return config
