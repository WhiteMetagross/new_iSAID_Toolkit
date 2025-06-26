import cv2
import os
import numpy as np
from natsort import natsorted
from glob import glob
from shutil import copyfile
import argparse

def main(args):
    src_root = args.src
    tar_root = args.tar
    splits = args.set.split(',')
    subfolder = args.image_sub_folder
    patch_h, patch_w = args.patch_height, args.patch_width
    overlap = args.overlap_area
    exts = ['.png', '.jpg', '.jpeg']

    for split in splits:
        if split not in ('train', 'val', 'test'):
            print(f"Skipping invalid split: {split}")
            continue

        print(f"\n>> Processing split: {split}")
        
        src_dir = os.path.join(src_root, split, subfolder)
        dst_dir = os.path.join(tar_root, split, subfolder)

        os.makedirs(dst_dir, exist_ok=True)

        suffixes = ['']
        if split in ('train', 'val'):
            suffixes.extend(['_instance_color_RGB', '_instance_id_RGB'])

        base_ids = []
        if not os.path.exists(src_dir):
            print(f"  [ERROR] Source directory not found: {src_dir}")
            continue
        else:
            for ext in exts:
                for fpath in glob(os.path.join(src_dir, f'*{ext}')):
                    name = os.path.splitext(os.path.basename(fpath))[0]
                    if '_' not in name:
                        base_ids.append(name)
        base_ids = natsorted(list(set(base_ids)))
        
        print(f"Found {len(base_ids)} raw images in {src_dir}")

        for base in base_ids:
            for suf in suffixes:
                fpath = None
                for ext in exts:
                    candidate = os.path.join(src_dir, f"{base}{suf}{ext}")
                    if os.path.exists(candidate):
                        fpath = candidate
                        break

                if fpath is None:
                    if not (split == 'test' and suf != ''):
                         print(f"  [WARN] missing file: {base}{suf} (searched exts: {exts})")
                    continue

                img = cv2.imread(fpath)
                if img is None:
                    print(f"  [ERROR] could not read: {os.path.basename(fpath)}")
                    continue
                h, w = img.shape[:2]

                if h > patch_h and w > patch_w:
                    for y0 in range(0, h, patch_h - overlap):
                        for x0 in range(0, w, patch_w - overlap):
                            y1 = min(y0 + patch_h, h)
                            x1 = min(x0 + patch_w, w)
                            
                            final_y0 = y1 - patch_h
                            final_x0 = x1 - patch_w
                            
                            patch = img[final_y0:y1, final_x0:x1]
                            
                            out_ext = os.path.splitext(fpath)[1]
                            out_name = f"{base}_{final_y0}_{y1}_{final_x0}_{x1}{suf}{out_ext}"
                            out_path = os.path.join(dst_dir, out_name)
                            cv2.imwrite(out_path, patch)
                else:
                    out_name = os.path.basename(fpath)
                    copyfile(fpath, os.path.join(dst_dir, out_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting the iSAID Images')
    parser.add_argument('--src', default='./iSAID_dataset', type=str)
    parser.add_argument('--tar', default='./iSAID_patches', type=str)
    parser.add_argument('--image_sub_folder', default='images', type=str)
    parser.add_argument('--set', default="train,val,test", type=str)
    parser.add_argument('--patch_width', default=800, type=int)
    parser.add_argument('--patch_height', default=800, type=int)
    parser.add_argument('--overlap_area', default=200, type=int)
    args = parser.parse_args()
    main(args)