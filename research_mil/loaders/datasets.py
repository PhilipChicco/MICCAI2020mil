import os, glob, copy
import collections
import random, sys, math
from tqdm import tqdm
import numpy as np
import openslide
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from PIL import Image, ImageEnhance
Image.MAX_IMAGE_PIXELS = 1000000000

# some code borrowed and modified from MIL-Nature-Medicine-2019 paper:
# credit : https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019

class MILdataset(Dataset):

    def __init__(self, libraryfile=None,
                 transform=None,
                 mult=1,
                 level=2,
                 class_map={'normal': 0, 'msi': 1},
                 nslides=-1,
                 train=True):

        self.skip     = 32
        self.classmap = class_map
        self.nslides  = nslides
        self.train    = train

        if libraryfile:
            lib = torch.load(libraryfile)
            """ Format of the library before pre-processing
               {'class_name':  torch_files.append({
                    "slide": wsi_file,
                    "grid" : points,
                    "target" : class_id }),
                 'class_name': ......,
                  ......................
                }
            """
            print('Loaded | ', libraryfile, self.classmap)
            
            
            lib = self.preprocess(lib)
            
        else:
            raise ('Please provide a lib file.')
        

        self.slidenames = lib['slides']
        self.slides = []
        self.grid = []
        self.slideIDX = []
        self.slideLBL = []
        self.targets = lib['targets']

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write('Opening Slides : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.slides.append(openslide.OpenSlide(slide))
            # load coords (x,y)
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))
        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        self.transform = transform
        self.mode   = 0
        self.mult   = mult
        self.size   = int(np.round(256 * self.mult))
        self.level  = level
        self.resize = 512

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]
        self.new_targets = [self.targets[self.slideIDX[x]] for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
        self.new_targets = [x[2] for x in self.t_data]

    def makevaldata(self, idxs):
        v_data = {self.slideIDX[x]: {'coord': [], 'label': None} for x in idxs}
        for x in idxs:
            v_data[self.slideIDX[x]]['coord'].append(self.grid[x])
            v_data[self.slideIDX[x]]['label'] = self.targets[self.slideIDX[x]]

        self.v_data = [(i, v_data[i]['coord'], v_data[i]['label']) for i in v_data]

    def shuffledata(self):
        self.v_data = random.sample(self.v_data, len(self.v_data))
        self.new_targets = [x[2] for x in self.v_data]

    def norm_coord(self, coord):
        """
        Normalize the coordinate to be uniform per mpp.
        Coordinates center will be the same for different levels.
        recommended : lv 0 - 2. with multiplier 1 - 4 atleast.

        We assume the extracted coordinates are already centered with ref to mask;
        but we will yet center them based on the chosen multiplier and level.
        """
        x, y = coord
        m = int(2 ** self.level)
        x = int(int(int(x) - int(self.size * m) / 2))
        y = int(int(int(y) - int(self.size * m) / 2))
        return (x, y)

    def __getitem__(self, index):
        if self.mode == 0:
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            coord = self.norm_coord(self.grid[index])
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')

            if self.mult != 1:
                img = img.resize((256, 256), Image.NEAREST)

            if self.transform is not None:
                img = self.transform(img)

            return img, target

        elif self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.norm_coord(self.grid[index])
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)
                return img

        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            coord = self.norm_coord(coord)
            img = self.slides[slideIDX].read_region(coord, self.level, (self.size, self.size)).convert('RGB')

            if self.mult != 1:
                img = img.resize((256, 256), Image.NEAREST)

            if self.transform is not None:
                img = self.transform(img)
            return img, target

        elif self.mode == 3:
            out_imgs   = []
            out_labels = []

            slide_id, coords, target = self.v_data[index]

            for x_y in coords:
                x_y = self.norm_coord(x_y)
                img = self.slides[slide_id].read_region(x_y, self.level, (self.size, self.size)).convert('RGB')

                if self.mult != 1:
                    img = img.resize((256, 256), Image.NEAREST)

                if self.transform is not None:
                    img = self.transform(img)
                out_imgs.append(img), out_labels.append(target)

            return out_imgs, target, out_labels
        else:  # for bag level inference

            slide_id, coords, target = self.v_data[index]

            out_imgs   = torch.zeros((len(coords), 3, 256, 256))
            out_labels = torch.zeros((len(coords)))

            for i, x_y in enumerate(coords):
                x_y = self.norm_coord(x_y)
                img = self.slides[slide_id].read_region(x_y, self.level, (self.size, self.size)).convert('RGB')
                
                if self.mult != 1:
                    img = img.resize((256, 256), Image.NEAREST)
    
                if self.transform is not None:
                    img = self.transform(img)

                out_imgs[i]   = img
                out_labels[i] = target
            # images [N,K,C,H,W] labels [N,K] -> [4,32,3,256,256] -> [4*32,3,256,256]
            return out_imgs, target, out_labels

    def __len__(self):
        if self.mode == 0:
            return len(self.grid)
        elif self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        else:
            return len(self.v_data)

    def preprocess(self, lib, change_root=False,prev_root=None,new_root=None):
        """
            Change format of lib file to:
            {
                'slides' : [xx.tif,xx2.tif , ....],
                'grid'   : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)

            ## TO DO: Change the root folder.
        """
        slides = []
        grid = []
        targets = []
        class_names = [x for x in self.classmap]
        for i, cls_id in enumerate(class_names):
            slide_dicts = lib[cls_id]
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):

                if isinstance(slide['grid'], type(None)):
                    print("Skipped [grid]: ", os.path.split(slide['slide'])[-1], self.classmap[slide['target']])
                    continue
                
                if self.train:
                    if len(slide['grid']) < self.skip:
                        print("Skipped [k<]: ", os.path.split(slide['slide'])[-1], self.classmap[slide['target']])
                        continue

                slides.append(slide['slide'])
                grid.append(slide['grid'])
                targets.append(self.classmap[slide['target']])

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}

#####
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2.2)


if __name__ == '__main__':
    # SANITY CHECK
    import torchvision.transforms as transforms
    
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    "====================================================="
    lib   = "./research_mil/data/mss_msi/val_lib.pth" 
    split = 'val'
    
    dset   = MILdataset(libraryfile=lib, mult=2, level=0, transform=trans, class_map={'mss': 0}, nslides=1)
    dset.setmode(1)
    loader = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)
    dset.setmode(1)

    for idx, data in enumerate(loader):
       sys.stdout.write('Dry Run : [{}/{}]\r'.format(idx+1, len(loader.dataset)))
    print(len(loader.dataset))

    for idx, data in enumerate(loader):
        print((data[0]).size())
        imshow(data.data.cpu().squeeze(0))

        if idx == 5:
            break
