
import torch
import numpy as np
import logging
import datetime
import os

from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values = []
        self.counter = 0

    def append(self, val: float) -> None:
        self.values.append(val)
        self.counter += 1

    def val(self) -> float:
        return self.values[-1]

    def avg(self) -> float:
        return sum(self.values) / len(self.values)

    def last_avg(self) -> float:
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def print_network(net, show_net=False):
    """ Print network definition"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net) if show_net else print("")
    num_params = num_params / 1000000.0
    print("----------------------------")
    print("MODEL: {:.5f}M".format(num_params))
    print("----------------------------")

def get_logger(logdir):
    logger = logging.getLogger('research_mil')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

