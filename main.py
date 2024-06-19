import os
import glob
import time

import numpy as np
import torch
from skimage.transform import resize

from utils import get_model, load_nii, resample, inference, comb_img_and_masks, save_nii, save_results

cta_dir = '../segmentation/data/nii_test/img3'
mask_dir = '../segmentation/data/nii_test/mask3'
out_dir = '../segmentation/data/nii_test/out3'

def main(model_name, save_pred):
    model, th = get_model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ctas = glob.glob(os.path.join(cta_dir, '*'))
    ctas.sort()
    img_ids = []
    classification = np.zeros((len(ctas)), dtype=np.int8)
    dices = np.zeros((len(ctas)))    

    for i, cta_path in enumerate(ctas):
        t1 = time.time()
        cta_id = os.path.basename(cta_path).split('_')[0]
        img_ids.append(cta_id)
        img, img_header = load_nii(cta_path)
        img_orig_shape = img.shape
        img_sp = img_header.get_zooms()
        img = resample(img, img_sp)

        mask_paths = glob.glob(os.path.join(mask_dir, cta_id + '*'))
        assert len(mask_paths) < 2
        if len(mask_paths) == 1:
            mask, mask_header = load_nii(mask_paths[0])
            mask_sp = mask_header.get_zooms()
            mask = mask > 0
            mask = resample(mask, mask_sp, order=0)
        else:
            mask = None

        seg = inference(model, th, img, device)

        if mask is None:
            classification[i] = 0 if seg.sum() == 0 else 1
        else:
            dice = 2 * (seg * mask).sum() / (seg.sum() + mask.sum())
            classification[i] = 2 if dice > 0.1 else 3
            dices[i] = dice        

        os.makedirs(out_dir, exist_ok=True)
        if save_pred:
            img_final = resize(img, img_orig_shape, order=3)
            seg_final = resize(seg, img_orig_shape, order=0)
            img_to_save = comb_img_and_masks(img_final, seg_final, alpha=0.1)
            save_path = os.path.join(out_dir, cta_id + '.nii.gz')
            save_nii(img_to_save, save_path, img_header)
        t2 = time.time()
        print(cta_path, t2-t1)

    save_results(classification, img_ids, out_dir, model_name)

if __name__ == '__main__':
    main('medswarm', True)