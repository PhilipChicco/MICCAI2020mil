
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy, numpy as np

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from models import get_model_bag
from utils.misc import print_network
from trainers.global_trainer import GlobalTrainer
from loaders import get_mildataset
from utils.mil import group_argtopk, group_max, calc_err
from utils.misc import AverageMeter



class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.device      = device

        center_init = torch.zeros(self.num_classes, self.feat_dim).to(self.device)

        nn.init.xavier_uniform_(center_init)
        self.centers = nn.Parameter(center_init)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long() # should be long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask   = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def get_assignment(self, batch):
        alpha = 1.0
        norm_squared = torch.sum((batch.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


class GlobalTaskMIL(GlobalTrainer):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                #transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        self.nclass = self.cfg['arch']['n_classes']

        loaders_dict = get_mildataset(self.cfg, self.data_transforms)
        self.train_dset, self.train_loader = loaders_dict['train']
        self.val_dset,   self.val_loader   = loaders_dict['val']

        finetuned_params = list(self.model.module.features.parameters())
        new_params = [p for n, p in self.model.module.named_parameters()
                      if not n.startswith('features.')]

        param_groups = [{'params': finetuned_params, 'lr': self.cfg['training']['lr']},
                        {'params': new_params,       'lr': 0.01 }]

        self.optimizer = optim.Adam(param_groups, weight_decay=1e-4)

        self.center_loss   = CenterLoss(num_classes=2,
                                        feat_dim=self.cfg['arch']['embedding'],
                                        device=self.device)

        # change sgd to Adam
        self.optimizerpool = optim.Adam(self.center_loss.parameters(),
                                    lr=self.cfg['training']['glr'])

        print()
        print(self.model.module.pooling)
        print()

    def _train_epoch(self, epoch):
        logits_losses  = AverageMeter()
        bag_losses     = AverageMeter()
        center_losses  = AverageMeter()
        train_accuracy = AverageMeter()
        bag_accuracy   = AverageMeter()

        self.center_loss.train()
        self.model.train()
        self.model.module.mode = 1 # combined mode (instance & bag prediction)

        self.adjust_lr_staircase(
            self.optimizer.param_groups,
            [0.001, 0.01], # initial values for features and pooling 
            epoch + 1,
            [10,15,17], # set the steps to adjust accordingly
            0.1 # reduce by this value
        )
        pbar = tqdm(self.train_loader, ncols=160, desc=' ')
        for i, (inputs, labels, all_labels) in enumerate(pbar):

            inputs     = inputs.to(self.device)
            labels     = labels.to(self.device)
            all_labels = all_labels.view(-1).to(self.device).long()

            self.optimizer.zero_grad()
            self.optimizerpool.zero_grad()

            # get features and logits
            inst_logits, inst_feat, bag_embed, bag_logits = self.model(inputs)

            loss_soft    = self.model.module.pooling.loss(inst_logits, all_labels)
            loss_bag     = self.model.module.pooling.loss(bag_logits, labels)

            # default : clustering instances
            #loss_center = self.center_loss(inst_embed, all_labels)
            # other : clustering bags / instances
            loss_center = self.center_loss(bag_feat, labels)
            # alpha, lambda and bag weight
            loss        = 1.0 * loss_soft + loss_center * 1.0 + loss_bag * 1.0


            preds_bag    = self.model.module.pooling.predictions(bag_logits)
            preds        = self.model.module.pooling.predictions(inst_logits)
            accuracy     = (preds == all_labels).sum().item() / all_labels.shape[0]
            accuracy_bag = (preds_bag == labels).sum().item() / labels.shape[0]

            loss_cen = loss_center.item()
            loss_val = loss_soft.item()
            loss_slide = loss_bag.item()
            logits_losses.append(loss_val)
            center_losses.append(loss_cen)
            bag_losses.append(loss_slide)
            train_accuracy.append(accuracy)
            bag_accuracy.append(accuracy_bag)

            loss.backward()
            self.optimizer.step()
            for param in self.center_loss.parameters():
                # center loss weight should match as in the loss function
                param.grad.data *= (1. / 1.0)
            self.optimizerpool.step()

            pbar.set_description(
                '--- (train) | Loss(I): {:.4f} | Loss(C): {:.4f} | Loss(B): {:.4f} | ACC(I): {:.3f} | ACC(B): {:.3f} :'.format(
                    logits_losses.avg(),
                    center_losses.avg(),
                    bag_losses.avg(),
                    train_accuracy.avg(),
                    bag_accuracy.avg()
                    )
            )

        step = epoch + 1
        self.writer.add_scalar('training/loss_i', logits_losses.avg(), step)
        self.writer.add_scalar('training/loss_c', center_losses.avg(), step)
        self.writer.add_scalar('training/loss_b', bag_losses.avg(), step)
        self.writer.add_scalar('training/accuracy', train_accuracy.avg(), step)
        self.writer.add_scalar('training/accuracy_bag', bag_accuracy.avg(), step)
        print()

    def _valid_epoch(self, epoch):
        self.val_dset.setmode(0) # 1 for max-max, 0 for max-min
        probs = self.inference(self.val_loader, self.cfg['training']['val_batch_size'], False)
        topk = group_argtopk(np.array(self.val_dset.slideIDX), probs,
                             self.cfg['training']['k'])
        self.val_dset.makevaldata(topk)
        self.val_dset.setmode(4)

        maxs = self.inference_aggregation()
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err, fpr, fnr = calc_err(pred, self.val_dset.targets)
        err = 1 - ((fpr + fnr) / 2.)

        # log
        self.writer.add_scalar('validation/accuracy', err, epoch)
        self.writer.add_scalar('validation/fpr', fpr, epoch)
        self.writer.add_scalar('validation/fnr', fnr, epoch)
        print()
        print('---> | accuracy : {:.4f}'.format(err))

        return {'acc': err,'fpr': fpr, 'fnr': fnr }

    def _on_train_start(self):

        self.train_dset.setmode(0)
        probs = self.inference(self.train_loader,
                               self.cfg['training']['train_batch_size'], True)
        topk = group_argtopk(np.array(self.train_dset.slideIDX), probs, self.cfg['training']['k'])
        self.train_dset.makevaldata(topk)
        self.train_dset.shuffledata()
        self.train_dset.setmode(4)

    def inference(self, loader, batch_size, train=False):
        self.model.eval()
        self.model.module.mode = 0
        probs = torch.FloatTensor(len(loader.dataset))

        with torch.no_grad():
            final_itr = tqdm(loader, ncols=80, desc='Inference(topk)...')

            for i, (input, targets) in enumerate(final_itr):
                input  = input.to(self.device)

                logits = self.model(input)[0]
                output = self.model.module.pooling.probabilities(logits)

                if self.nclass == 2:
                     probs[i * batch_size: i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
                else:  # when using deepmil
                     probs[i * batch_size: i * batch_size + input.size(0)] = (output.detach()).clone()

        return probs.cpu().numpy()

    def inference_aggregation(self):
        self.model.eval()
        self.center_loss.eval()
        self.model.module.mode = 1
        probs = torch.FloatTensor(len(self.val_loader.dataset))

        with torch.no_grad():
            final_itr = tqdm(self.val_loader, ncols=80, desc='Inference (topk-Aggregation) ...')

            for i, data in enumerate(final_itr):
                input  = data[0]
                input  = input.to(self.device)
                
                ###
                # FOR SOFT-ASSIGNMENT BASED INFERENCE
                # USE BAG NORM. FEATURE TO GET PREDICTED LABEL
                # logits   = self.model(input)[2]
                # output   = self.center_loss.get_assignment(logits)
                # ITS ALREADY A PROB DISTRIBUTION, NO NEED FOR pooling.probabilities(output)
                # probs[i] = output.detach()[:, 1].clone()
                ###
                
                # USING BAG CLASSIFIER
                logits   = self.model(input)[3]
                output   = self.model.module.pooling.probabilities(logits)
                probs[i] = output.detach()[:, 1].clone()
                #####

        return probs.cpu().numpy()

    # not used
    def _on_train_end(self):
        pass

    def _on_valid_start(self):
        pass

    def _on_valid_end(self):
        pass

    def _on_model_state(self):
        m_weights = copy.deepcopy(self.model.state_dict())
        c_weights = copy.deepcopy(self.center_loss.state_dict())
        state = {'model_state': m_weights, 'center_state': c_weights }
        return state

    def find_index(self,seq, item):
        for i, x in enumerate(seq):
            if item == x:
                return i
        return -1

    def adjust_lr_staircase(self, param_groups, base_lrs, ep, decay_at_epochs=[1, 2], factor=0.1):
        """Multiplied by a factor at the BEGINNING of specified epochs. Different
        param groups specify their own base learning rates.

        Args:
          param_groups: a list of params
          base_lrs: starting learning rates, len(base_lrs) = len(param_groups)
          ep: current epoch, ep >= 1
          decay_at_epochs: a list or tuple; learning rates are multiplied by a factor
            at the BEGINNING of these epochs
          factor: a number in range (0, 1)

        Example:
          base_lrs = [0.1, 0.01]
          decay_at_epochs = [51, 101]
          factor = 0.1
          It means the learning rate starts at 0.1 for 1st param group
          (0.01 for 2nd param group) and is multiplied by 0.1 at the
          BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
          BEGINNING of the 101'st epoch, then stays unchanged till the end of
          training.

        NOTE:
          It is meant to be called at the BEGINNING of an epoch.
        """
        assert len(base_lrs) == len(param_groups), \
            "You should specify base lr for each param group."
        assert ep >= 1, "Current epoch number should be >= 1"

        if ep not in decay_at_epochs:
            return

        ind = self.find_index(decay_at_epochs, ep)

        for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
            g['lr'] = base_lr * factor ** (ind + 1)
            print('===> Param group {}: lr adjusted to {:.10f}'.format(i, g['lr']).rstrip('0'))
