
import torch, os
import numpy as np

from tqdm import tqdm
from torchvision import transforms

from evaluators.global_tester import GlobalTester
from loaders import get_mildataset_test
from utils.mil import group_argtopk, group_max, calc_err
from utils.misc import AverageMeter
from trainers.globaltaskmil_trainer import CenterLoss

class GlobalTaskMILTest(GlobalTester):

    def __init__(self, cfg, logdir, logger):
        super().__init__(cfg, logdir, logger)

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
            ])
        }

        self.nclass = self.cfg['arch']['n_classes']
        self.test_dset, self.test_loader = get_mildataset_test(self.cfg, self.data_transforms)['test']

        self.center_loss = CenterLoss(num_classes=self.nclass,
                                      feat_dim=self.cfg['arch']['embedding'],
                                      device=self.device)
        # load state
        chk = torch.load(self.ckpt)
        state = chk["center_state"]
        self.center_loss.load_state_dict(state)
        self.center_loss.eval()
        print(self.center_loss)

        print()

    def _run_test(self):
        self.test_dset.setmode(0)
        probs = self.inference()
        topk = group_argtopk(np.array(self.test_dset.slideIDX), probs,
                             self.cfg['training']['k'])
        self.test_dset.makevaldata(topk)
        self.test_dset.setmode(4)

        maxs = self.inference_aggregation()
        fp = open(os.path.join(self.logdir, 'predictions.csv'), 'w')
        fp.write('file,target,prediction,probability\n')
        for name, target, prob in zip(self.test_dset.slidenames, self.test_dset.targets, maxs):
            fp.write('{},{},{},{:.3f}\n'.format(os.path.split(name)[-1], target, int(prob >= 0.5), prob))
        fp.close()
        
        # check accuracy
        pred = [1 if x >= 0.5 else 0 for x in maxs]
        err, fpr, fnr = calc_err(pred, self.test_dset.targets)
        err = 1 - ((fpr + fnr) / 2.)
        print('--- (test)[AGG] | Accuracy: {:.3f} | FPR: {:.3f} | FNR: {:.3f} '.format(
            err, fpr, fnr
        ))

        # For instance classification
        # self.inference_classification()
        

    def inference(self):
        self.model.eval()
        self.model.module.mode = 0
        probs = torch.FloatTensor(len(self.test_loader.dataset))

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference(topk)...')

            for i, (input,_) in enumerate(final_itr):
                input  = input.to(self.device)
                logits = self.model(input)[0]
                output = self.model.module.pooling.probabilities(logits)

                if self.nclass == 2:
                    probs[i] = output.detach()[:, 1].clone()
                else:  # using deepmil
                    probs[i] = (output.detach()).clone()

        return probs.cpu().numpy()

    def inference_aggregation(self):
        self.model.eval()
        self.model.module.mode = 1
        probs = torch.FloatTensor(len(self.test_loader.dataset))

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference (Aggregation) ...')

            for i, data in enumerate(final_itr):
                input = data[0]
                input = input.to(self.device)

                # Soft Assignment based inference
                # Comment this to use bag classifier below
                # logits   = self.model(input)[2]
                # output   = self.center_loss.get_assignment(logits)
                # probs[i] = output.detach()[:, 1].clone()

                # Bag classifier based inference
                # Comment this to use soft-assignment above
                logits   = self.model(input)[3]
                output   = self.model.module.pooling.probabilities(logits)
                probs[i] = output.detach()[:, 1].clone()

        return probs.cpu().numpy()

    def inference_classification(self):
        self.model.eval()
        self.model.module.mode = 0
        val_accuracy = AverageMeter()

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference (instance) ...')

            for i, (input, labels) in enumerate(final_itr):
                input  = input.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input)[0]
                preds  = self.model.module.pooling.predictions(logits)

                accuracy = (preds == labels).sum().item() / labels.shape[0]
                val_accuracy.append(accuracy)

                final_itr.set_description('--- (test) | Accuracy: {:.3f}  :'.format(
                    val_accuracy.avg())
                )

        err = val_accuracy.avg()
        fp = open(os.path.join(self.logdir, 'meanscores.csv'), 'w')
        fp.write('Accuracy: {:.4f} \n'.format(err))
        fp.close()


    # not used
    def _on_test_end(self):
        pass

    def _on_test_start(self):
        pass