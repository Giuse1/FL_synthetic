import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  config.total_num_users = 100
  config.users_per_round = 50

  config.num_rounds = 1000
  config.local_epochs = 1

  config.optimizer = "SGD"
  config.batch_size = 64  # 16
  config.learning_rate = 0.001 * 5
  config.decay = 0.999

  config.mode = "iid"
  config.model = "resnet-19"

  config.dataset = "cifar10"
  print(f"total_num_users: {config.total_num_users}")
  print(f"users_per_round: {config.users_per_round}")
  print(f"num_rounds: {config.num_rounds}")
  print(f"local_epochs: {config.local_epochs}")
  print(f"optimizer: {config.optimizer}")
  print(f"batch_size: {config.batch_size}")
  print(f"learning_rate: {config.learning_rate}")
  print(f"decay: {config.decay}")
  print(f"mode: {config.mode}")
  print(f"model: {config.model}")

  return config