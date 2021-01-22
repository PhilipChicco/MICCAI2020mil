## Refer to MetricMWSI and SemiSupervised Repos

import warnings
warnings.filterwarnings("ignore")

import argparse
import yaml
import os, torch, random, numpy as np

# trainers
from trainers import get_trainer
from utils.misc import get_logger

from tensorboardX import SummaryWriter

def main(cfg):
    print(cfg)
    print()

    # setup logdir, writer and logger
    logdir = os.path.join(cfg['root'], cfg['logdir'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    trainer_name = cfg['trainer']

    with open(os.path.join(logdir,trainer_name+'.yml'), 'w') as fp:
        yaml.dump(cfg, fp)

    logger  = get_logger(logdir)

    Trainer = get_trainer(trainer_name)(cfg, writer, logger)
    print()

    # start training
    Trainer.train()



if __name__ == '__main__':

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # get configs
    parser = argparse.ArgumentParser(description="Train a Network")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="research_mil/configs/globalloss_mil_train.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    main(cfg)