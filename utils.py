import os

import torch
from monai.networks.nets import SwinUNETR as MonaiSwin

from mednext import MedNeXt

THRESHOLDS = {'medcent':0,
             'medswarm':0,
             'swincent':0,
             'swinswarm':0}

def get_model(model_name):
    if 'med' in model_name:
        model = MedNeXt(in_channels=1, n_channels=32, n_classes=2, exp_r=2, block_counts=[2,2,2,2,2,2,2,2,2], dim='3d')
    else:
        model = MonaiSwin(img_size=[128, 128, 128], in_channels=1, out_channels=2, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], feature_size=48, spatial_dims=3)
    
    state_dict = torch.load(os.path.join('state_dicts', model_name + '.state'))
    model.load_state_dict(state_dict)
    return model

def load_dcm(path):
    pass

def load_mask(path):
    pass

def inference(model, th, img):
    pass

def comb_img_and_masks(img, pred, mask):
    pass

def save_nii(array, path):
    pass