import os
import csv
from functools import lru_cache

import numpy as np
from scipy import ndimage
import torch
import nibabel as nib
from skimage.transform import resize
from monai.networks.nets import SwinUNETR as MonaiSwin

from mednext import MedNeXt

THRESHOLDS = np.load('data/thresholds.npy', allow_pickle=True).item()
PREPROC = np.load('data/preproc.npy', allow_pickle=True).item()

def get_model(model_name):
    if 'med' in model_name:
        model = MedNeXt(in_channels=1, n_channels=32, n_classes=2, exp_r=2, block_counts=[2,2,2,2,2,2,2,2,2], dim='3d')
    else:
        model = MonaiSwin(img_size=[128, 128, 128], in_channels=1, out_channels=2, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], feature_size=48, spatial_dims=3)
    
    state_dict = torch.load(os.path.join('data', model_name + '.state'))['model_state']
    state_dict = {key.replace('module.', ''):value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    eps = 1e-6
    return model, THRESHOLDS[model_name] + eps

def load_nii(nii_path):
    nii = nib.load(nii_path)
    header = nii.header.copy()
    vol = nii.get_fdata()
    return vol, header

def resample(vol, spacing, order=3):
    orig_shape = vol.shape
    target_spacing = np.array([1., 1., 1.])

    final_shape = np.array([int(round(i / j * k)) for i, j, k in zip(spacing, target_spacing, orig_shape)])
    resampled = resize(vol, final_shape, order=order, preserve_range=True)
    
    if order > 0:
        resampled = np.clip(resampled, a_min=PREPROC['min'], a_max=PREPROC['max'])
        resampled = (resampled - PREPROC['mean']) / PREPROC['std']
        
    return resampled

@lru_cache(maxsize=2)
def compute_gaussian(tile_size : tuple, sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device='cuda:0') \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = ndimage.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.to(dtype).to(device)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def postprocess_to_single_comp(pred_class):
    if pred_class.sum() > 0:
        labelled_components, num_components = ndimage.label(pred_class)
        comp_sizes = np.zeros((num_components))
        for comp in range(1, num_components+1):
            comp_sizes[comp-1] = (labelled_components == comp).sum()
        largest_comp = comp_sizes.argmax() + 1
        post = labelled_components == largest_comp
    else:
        post = np.zeros_like(pred_class)

    return post

def inference(model, th, img, device):
    img = np.flip(img.transpose(1, 0, 2), axis=0).copy()
    img = torch.from_numpy(img).float()
    model = model.to(device)
    model.eval()
    H, W, D = img.shape
    pred = torch.zeros((2, H, W, D), device='cpu')
    n_predictions = torch.zeros((H, W, D), device='cpu')
    gaussian = compute_gaussian(tile_size=(128, 128 , 128), device='cpu')

    x, y, z = [128, 128, 128]
    x_step, y_step, z_step = [int(x*0.5), int(y*0.5), int(z*0.5)]
    x_coords = list(range(0, H-x, x_step))
    x_coords.append(H-x)
    y_coords = list(range(0, W-y, y_step))
    y_coords.append(W-y)
    z_coords = list(range(0, D-z, z_step))
    z_coords.append(D-z)

    for x_c in x_coords:
        for y_c in y_coords:
            for z_c in z_coords:
                patch = img[x_c: x_c+x, y_c: y_c+y, z_c: z_c+z]
                patch = patch.to(device).unsqueeze(0).unsqueeze(0)
                temp1 = model(patch)
                temp1 = temp1.squeeze().detach().cpu()
                pred[:, x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += temp1 * gaussian
                n_predictions[x_c: x_c+x, y_c: y_c+y, z_c: z_c+z] += gaussian
                del temp1, patch
    pred /= n_predictions
    pred = pred.numpy()

    e_pred = np.exp(pred - pred.max(axis=0))
    prob = (e_pred / np.sum(e_pred, axis=0, keepdims=True))[1]
    post = postprocess_to_single_comp(prob > th)
    post = np.flip(post, axis=0).transpose(1, 0, 2)

    return post

def comb_img_and_masks(img, mask, alpha=0.3):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    mask = mask > 0
    img[mask] = img[mask]*alpha + (1 - alpha) * 255
    return img.astype(dtype=np.uint8)

def save_nii(array, path, header):
    nii = nib.Nifti1Image(array, None, header=header)
    nib.save(nii, path)

def save_results(cl, ids, out_dir, model_name):
    tp = (cl == 2).sum()
    fp = (cl == 1).sum()
    tn = (cl == 0).sum()
    fn = (cl == 3).sum()
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    npv = tn / (tn + fn) if tn + fn > 0 else 0
    f1 = 2*tp / (2*tp + fn + fp) if 2*tp + fn + fp > 0 else 0
    f2 = 5*tp / (5*tp + 4*fn + fp) if 5*tp + 4*fn + fp > 0 else 0

    res_dict = {'f1':f1,
                'f2':f2,
                'sensitivity':sensitivity,
                'specificity':specificity,
                'ppv':ppv,
                'npv':npv,
                'tp ids':[ids[i] for i in np.where(cl == 2)[0]],
                'fp ids':[ids[i] for i in np.where(cl == 1)[0]],
                'tn ids':[ids[i] for i in np.where(cl == 0)[0]],
                'fn ids':[ids[i] for i in np.where(cl == 3)[0]]}
    np.save(os.path.join(out_dir, f'{model_name}_results.npy'), res_dict)
    
    with open(os.path.join(out_dir, f'{model_name}_results.csv'), 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        for key, value in res_dict.items():
            if 'ids' in key:
                row = [key] + value
            else:
                row = [key, value]
            wr.writerow(row)