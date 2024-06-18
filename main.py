import os
import glob

from utils import get_model, load_dcm, load_mask, inference, comb_img_and_masks, save_nii, THRESHOLDS

cta_dir = ''
mask_dir = ''
out_dir = None

def main(model_name, save_pred):
    model = get_model(model_name)
    th = THRESHOLDS[model_name]

    ctas = glob.glob(cta_dir)
    ctas.sort()

    for cta_path in ctas:
        cta_id = os.path.basename(cta_path)
        img = load_dcm(cta_path)
        pred = inference(model, th, img)

        mask_path = glob.glob(os.path.join(mask_dir, cta_id + '*'))[0]
        mask = load_mask(mask_path)

        dice = 2 * (pred * mask).sum() / (pred.sum() + mask.sum())
        # report metric???
        if save_pred:
            # resample img and mask???
            img_to_save = comb_img_and_masks(img, pred, mask)
            save_path = os.path.join(out_dir, cta_id + '.nii')
            save_nii(img_to_save, save_path)


if __name__ == '__main__':
    main()