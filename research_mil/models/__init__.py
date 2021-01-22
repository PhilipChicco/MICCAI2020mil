
import torch, sys, os
import torch.nn as nn
from collections import OrderedDict

from models.backbones import load_backbone
from models.pooling import load_pooling
from models.pooling.methods import BagNetwork

# other type of models
def get_model(cfg):
    
    backbone  = cfg['arch']['backbone']
    pooling   = cfg['arch']['pooling']
    n_classes = cfg['arch']['n_classes']
    embed     = cfg['arch']['embedding']

    backbone     = load_backbone(backbone)
    out_channels = backbone.inplanes

    pooling = load_pooling(pooling, out_channels, n_classes, embed)

    # final model
    model = nn.Sequential(OrderedDict([
        ('features', backbone),
        ('pooling' , pooling)
    ]))

    return model

# proposed model loader
def get_model_bag(cfg):
    backbone  = cfg['arch']['backbone']
    pooling   = cfg['arch']['pooling']
    n_classes = cfg['arch']['n_classes']
    embed     = cfg['arch']['embedding']

    backbone = load_backbone(backbone)
    out_channels = backbone.inplanes

    pooling = load_pooling(pooling, out_channels, n_classes, embed)

    # final model
    bagModel = BagNetwork(backbone, pooling, n_classes, embed, k=cfg['training']['k'])

    return bagModel



