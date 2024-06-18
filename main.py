import os
import glob

import torch

from utils import get_model, load_nii, resample, inference, comb_img_and_masks, save_nii

cta_dir = ''
mask_dir = ''
out_dir = None

def main(model_name, save_pred):
    model, th = get_model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ctas = glob.glob(cta_dir)
    ctas.sort()

    for cta_path in ctas:
        cta_id = os.path.basename(cta_path)
        img, img_header = load_nii(cta_path)
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
            mask == None

        seg = inference(model, th, img, device)


        dice = 2 * (seg * mask).sum() / (seg.sum() + mask.sum())
        # report metric???
        if save_pred:
            # resample img and mask???
            img_to_save = comb_img_and_masks(img, seg, mask)
            save_path = os.path.join(out_dir, cta_id + '.nii.gz')
            save_nii(img_to_save, save_path)


if __name__ == '__main__':
    main()