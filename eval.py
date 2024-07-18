import argparse
import os
import csv

from config import eval_config
from utils import load_nii

def eval(pred_file, mask_file, out_dir):
    pred, _ = load_nii(pred_file)
    mask, _ = load_nii(mask_file)
    assert pred.shape == mask.shape
    intersection = (pred * mask).sum() > 0

    if intersection:
        print('The ground truth and prediction intersect. True positive prediction.')
    else:
        print('The ground truth and prediction do not intersect. False negative prediction.')

    with open(os.path.join(out_dir, f'{pred_file}_results.csv'), 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow([pred_file, mask_file, intersection])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Occlusion validation')
    parser.add_argument('--pred_file', help='Filename of predicted mask', type=str, default=None)
    parser.add_argument('--mask_file', help='Filename of ground truth mask', type=str, default=None)
    parser.add_argument('--out_dir', help='Directory for output. Created if does not already exist', type=str, default=None)
    args = parser.parse_args()
    print(args)
    eval_config['pred_file'] = args.pred_file if args.pred_file is not None else eval_config['pred_file']
    eval_config['mask_file'] = args.mask_file if args.mask_file is not None else eval_config['mask_file']
    eval_config['out_dir'] = args.out_dir if args.out_dir is not None else eval_config['out_dir']
    print(**eval_config)
    eval()