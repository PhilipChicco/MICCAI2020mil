
import torch
import torch.backends.cudnn as cudnn
from abc import abstractmethod

import os, random
import copy, time
import numpy as np


from models import get_model
from utils.misc import print_network

class BaseTrainer:

    def __init__(self, cfg, writer, logger):

        self.cfg    = cfg
        self.writer = writer
        self.logdir = self.writer.file_writer.get_logdir()
        self.logger = logger

        # setup seeds
        self._setup_seeds()

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(self.cfg['n_gpu'])
        print(self.device, self.device_ids)
        # get model
        # add checks for resume training
        model = get_model(cfg)
        self.model = self.setup_models(model)

        self.num_epochs  = self.cfg['training']['epochs']
        self.monitor     = self.cfg['training']['monitor']
        self.resume_name = self.cfg['training']['resume']

        cudnn.benchmark = True

    def setup_models(self, model):
        model = model.to(self.device)
        # load states here
        if len(self.device_ids) >= 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        print_network(model)
        return model

    def _setup_seeds(self):
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        random.seed(self.cfg['seed'])

    def train(self):

        self.best_acc = 0.0

        since = time.time()
        for epoch in range(self.num_epochs):
            # epoch
            print('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._on_train_start()
                    self.model.train()
                    self._train_epoch(epoch)
                    self._on_train_end()
                else:
                    self.model.eval()
                    self._on_valid_start()
                    results = self._valid_epoch(epoch)
                    self._on_valid_end()
                    monitor_val = results[self.monitor]

                    if monitor_val >= self.best_acc:
                        print(' |--(saved model)-- best {}: {:.3f}\n'.format(self.monitor,monitor_val))
                        self.best_acc = monitor_val
                        self.on_save_models(self._on_model_state())

        # done !!
        time_elapsed = time.time() - since
        print('Training complete | {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
        "============================================================================================="

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _on_train_start(self):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _on_train_end(self):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_model_state(self):
        weights = copy.deepcopy(self.model.state_dict())
        state = {'model_state': weights, }
        return state

    def on_save_models(self, state):
        save_path = os.path.join(self.logdir, "{}".format(self.resume_name))
        torch.save(state, save_path)

    @abstractmethod
    def _on_valid_start(self):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _on_valid_end(self):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

