
import torch
import torch.backends.cudnn as cudnn
from abc import abstractmethod

import os, random
import copy, time
import numpy as np

from models import get_model
from utils.misc import print_network, convert_state_dict



class BaseTester:

    def __init__(self, cfg, logdir, logger):

        self.cfg      = cfg
        self.logdir   = logdir
        self.logger   = logger
        self.ckpt     = os.path.join(self.cfg['root'],self.cfg['testing']['checkpoint'])

        # setup seeds
        self._setup_seeds()

        # load model(s)
        self.device_ids = [0]
        self.device = torch.device('cuda:0' if self.cfg['n_gpu'] > 0 else 'cpu')
        self.model  = self._setup_models(self.cfg)


    def _setup_models(self, cfg):
        model = get_model(cfg)
        self.model = model.to(self.device)

        if self.cfg['trainer'] == 'deepset' or self.cfg['trainer'] == 'deepsetweak':
            self.model.features = torch.nn.Identity().to(self.device)

        if os.path.exists(self.ckpt):
            model_chk  = torch.load(self.ckpt)
            state = convert_state_dict(model_chk["model_state"])
            self.model.load_state_dict(state)
            print('Loaded | ',self.ckpt)
        # load states here
        if len(self.device_ids) >= 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        print_network(self.model)
        return self.model

    def test(self):

        since = time.time()
        self._on_test_start()
        self._run_test()
        self._on_test_end()
        time_elapsed = time.time() - since

        print('Testing complete | {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
        "==========================================================================================="

    def _setup_seeds(self):
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        random.seed(self.cfg['seed'])

    def visualize_features(self):
        pass

    @abstractmethod
    def _run_test(self):
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _on_test_start(self):
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _on_test_end(self):
        """
        """
        raise NotImplementedError