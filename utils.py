import logging


def setup_logger(name, log_file, isServer, level=logging.INFO):

        handler = logging.FileHandler(log_file, mode='w')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        if isServer:
            logger.info("round,phase,loss_real,acc_real,loss_syn,acc_syn")
        else:
            logger.info("round,phase,loss,correct,total_samples")
        return logger
