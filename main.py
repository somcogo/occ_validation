import os
import glob
import time
import argparse

import numpy as np
import torch
from skimage.transform import resize

from config import config
from utils import get_model, load_nii, resample, inference, comb_img_and_masks, save_nii, save_results

def main():
    model, th = get_model(config['model_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ctas = glob.glob(os.path.join(config['cta_dir'], '*'))
    ctas.sort()
    img_ids = []
    classification = np.zeros((len(ctas)), dtype=np.int8)
    dices = np.zeros((len(ctas)))    

    for i, cta_path in enumerate(ctas):
        t1 = time.time()
        cta_id = os.path.basename(cta_path).replace('.nii.gz', '').replace('.nii', '')

        mask_paths = glob.glob(os.path.join(config['mask_dir'], cta_id + '*'))
        try:
            assert len(mask_paths) < 2
        except AssertionError:
            print(f'More than one mask was found for patient id {cta_id} so the scan is ignored, the mask paths are:', mask_paths)
            continue

        img_ids.append(cta_id)
        img, img_header = load_nii(cta_path)
        img_orig_shape = img.shape
        img_sp = img_header.get_zooms()
        img = resample(img, img_sp)
        if len(mask_paths) == 1:
            mask, mask_header = load_nii(mask_paths[0])
            mask_sp = mask_header.get_zooms()
            mask = mask > 0
            mask = resample(mask, mask_sp, order=0)
        else:
            mask = None

        seg, prob = inference(model, th, img, device)

        os.makedirs(config['out_dir'], exist_ok=True)

        if mask is None:
            classification[i] = 0 if seg.sum() == 0 else 1
        else:
            dice = 2 * (seg * mask).sum() / (seg.sum() + mask.sum())
            classification[i] = 2 if dice > 0.1 else 3
            dices[i] = dice        

        print('The value of no_save is:', config['no_save'])
        if not config['no_save']:
            img_final = resize(img, img_orig_shape, order=3)
            seg_final = resize(seg, img_orig_shape, order=0)
            img_to_save = comb_img_and_masks(img_final, seg_final, alpha=0.1)
            save_path = os.path.join(config['out_dir'], cta_id + '.nii.gz')
            save_nii(img_to_save, save_path, img_header)
        t2 = time.time()
        print(str(i+1)+'/'+str(len(ctas)), cta_path, t2-t1)

    save_results(classification, img_ids, config['out_dir'], config['model_name'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Occlusion validation')
    parser.add_argument('--no_save', help='If true, model prediction is save to out_dir', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--model_name', help='Name of the model used to make predictions', type=str, default='medcent')
    parser.add_argument('--cta_dir', help='Path to CTA scans', type=str, default=None)
    parser.add_argument('--mask_dir', help='Path to masks', type=str, default=None)
    parser.add_argument('--out_dir', help='Directory for output. Created if does not already exist', type=str, default=None)
    args = parser.parse_args()
    print(args)
    config['no_save'] = args.no_save
    config['model_name'] = args.model_name
    config['cta_dir'] = args.cta_dir if args.cta_dir is not None else config['cta_dir']
    config['mask_dir'] = args.mask_dir if args.mask_dir is not None else config['mask_dir']
    config['out_dir'] = args.out_dir if args.out_dir is not None else config['out_dir']
    assert config['model_name'] in ['medcent', 'medswarm', 'swincent', 'swinswarm']
    print(config)
    main()