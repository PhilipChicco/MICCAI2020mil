import os, torch
from torch.utils.data import DataLoader
from loaders.datasets import MILdataset
from . import *


loaders = {
    "milLoader"      : MILdataset,
}


def get_mildataset(cfg, data_transforms):

    class_map   = {x:idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path   = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        libraryfile=os.path.join(data_path, cfg['data']['train_split'] + '_lib.pth'),
        mult=cfg['data']['mult'],
        level=cfg['data']['level'],
        transform=data_transforms['train'], class_map=class_map, nslides=cfg['data']['train_nslides'])
    
    
    t_loader = DataLoader(t_dset,
            batch_size=cfg['training']['train_batch_size'],
            num_workers=cfg['training']['n_workers'],
            shuffle=False, # randomness is internally achieved in the loader
            pin_memory=True
    )


    v_dset = data_loader(
        libraryfile=os.path.join(data_path, cfg['data']['val_split'] + '_lib.pth'),
        mult=cfg['data']['mult'],
        level=cfg['data']['level'],
        transform=data_transforms['val'], class_map=class_map, nslides=cfg['data']['val_nslides'])
    v_loader = DataLoader(v_dset,
               batch_size=cfg['training']['val_batch_size'],
               num_workers=cfg['training']['n_workers'],
               shuffle=False, pin_memory=True
               )

    return {'train': (t_dset,t_loader), 'val': (v_dset,v_loader) }

def get_mildataset_test(cfg, data_transforms):

    class_map   = {x:idx for idx, x in enumerate(cfg['data']['classmap'].split(","))}
    data_path   = cfg['data']['data_path']
    data_loader = loaders[cfg['data']['dataset']]

    t_dset = data_loader(
        libraryfile=os.path.join(data_path, cfg['data']['test_split'] + '_lib.pth'),
        mult=cfg['data']['mult'],
        level=cfg['data']['level'],
        transform=data_transforms['test'], 
        class_map=class_map, 
        nslides=cfg['data']['test_nslides'],
        train=False)
    t_loader = DataLoader(t_dset,
               batch_size=cfg['training']['test_batch_size'],
               num_workers=cfg['training']['n_workers']//2,
               shuffle=False, pin_memory=False,
               )

    return {'test': (t_dset, t_loader) }


datamethods = {
    'milLoader'    : get_mildataset,
}
