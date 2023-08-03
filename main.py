import random

import numpy as np
import torch
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from FL import FL_train
from fedBN import fedBN_train
from fedBN import heteroFL_train

# todo local accuracy, when test on client
# todo batch normalization
# todo find reports/ ! -name '*.npz' -type f | xargs cp --parents -t reports/cifar10/to_copy_25_07
# todo  grep -H star *.py

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
# flags.DEFINE_string("model", None, "model.")
# flags.DEFINE_string("data_distribution", None, "data distribution.")
# flags.DEFINE_string("validation_node", None, "validation at no.")

flags.mark_flags_as_required(["config"])

def main(argv):

    SEED = FLAGS.config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    FLAGS.config.algorithm = "heteroFL"

    if FLAGS.config.model == "resnet18":
        if FLAGS.config.algorithm == "fedbn" or FLAGS.config.algorithm =="siloer":
            print(FLAGS.config.algorithm)
            fedBN_train.train_model(FLAGS.config)
        elif FLAGS.config.algorithm == "heteroFL":
            heteroFL_train.train_model(FLAGS.config)
        else:
            FL_train.train_model(FLAGS.config)

    else:
        FL_train.train_model(FLAGS.config)
    # elif mode == "standard_noniid": train_loss, train_acc, val_loss, val_acc = FL_train.train_model(model,
    # criterion, num_rounds=num_rounds, local_epochs=local_epochs, total_num_users=total_num_users,
    # num_users=num_users, batch_size=batch_size, learning_rate=learning_rate, decay=decay, iid=False)


if __name__ == "__main__":
    app.run(main)
