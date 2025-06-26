import argparse
import os
import json
import cv2
import numpy as np
from natsort import natsorted
from pycocotools import mask as maskUtils
from skimage import measure

def parse_args():
    parser = argparse.ArgumentParser(description='Convert iSAID patches to COCO-style JSON')
    parser.add_argument('--datadir', default='./iSAID_patches', type=str)
    parser.add_argument('--outdir', default='./iSAID_patches', type=str)
    parser.add_argument('--set', default="train,val", type=str)
    return parser.parse_args()

def get_category_info():
    return [
        {'id': 0, 'name': 'unlabeled'}, {'id': 1, 'name': 'ship'},
        {'id': 2, 'name': 'storage_tank'}, {'id': 3, 'name': 'baseball_diamond'},
        {'id': 4, 'name': 'tennis_court'}, {'id': 5, 'name': 'basketball_court'},
        {'id': 6, 'name': 'Ground_Track_Field'}, {'id': 7, 'name': 'Bridge'},
        {'id': 8, 'name': 'Large_Vehicle'}, {'id': 9, 'name': 'Small_Vehicle'},
        {'id': 10, 'name': 'Helicopter'}, {'id': 11, 'name': 'Swimming_pool'},
        {'id': 12, 'name': 'Roundabout'}, {'id': 13, 'name': 'Soccer_ball_field'},
        {'id': 14, 'name': 'plane'}, {'id': 15, 'name': 'Harbor'}
    ]

def main(args):
    categories = get_category_info()

    for split in args.set.split(','):
        print(f"Processing split: {split}")
        patch_dir = os.path.join(args.datadir, split, 'images')
        if not os.path.exists(patch_dir):
            print(f"Directory not found: {patch_dir}")
            continue

        images = []
        annotations = []
        ann_id = 0
        img_id = 0

        all_files = natsorted(os.listdir(patch_dir))
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_instance_' not in f]

        for img_file in image_files:
            base_name, img_ext = os.path.splitext(img_file)
            ins_file = f"{base_name}_instance_id_RGB.png"
            ins_path = os.path.join(patch_dir, ins_file)
            img_path = os.path.join(patch_dir, img_file)

            if not os.path.exists(ins_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            h, w, _ = img.shape

            images.append({
                'id': img_id,
                'width': w,
                'height': h,
                'file_name': img_file,
            })

            instance_img = cv2.imread(ins_path)
            if instance_img is None:
                print(f"Could not read instance image: {ins_path}")
                img_id += 1
                continue
            
            r_channel = instance_img[:, :, 2].astype(np.int32)
            g_channel = instance_img[:, :, 1].astype(np.int32)
            instance_map = (r_channel // 10 * 256) + g_channel
            
            unique_instances = np.unique(instance_map)

            for instance_id in unique_instances:
                if instance_id == 0: continue
                
                class_id = instance_id // 1000
                if class_id == 0 or class_id > len(categories) - 1: continue
                
                binary_mask = (instance_map == instance_id).astype(np.uint8)
                
                if binary_mask.sum() < 10: continue

                contours = measure.find_contours(binary_mask, 0.5)
                
                segmentation = []
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation.append(contour.ravel().tolist())
                
                if not segmentation: continue

                rle = maskUtils.encode(np.asfortranarray(binary_mask))
                area = float(maskUtils.area(rle))
                bbox = maskUtils.toBbox(rle).tolist()

                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': int(class_id),
                    'segmentation': segmentation,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                })
                ann_id += 1
            img_id += 1

        coco_dict = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        
        out_json_path = os.path.join(args.outdir, split, f'instancesonly_filtered_{split}.json')
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, 'w') as f:
            json.dump(coco_dict, f, indent=4)
        
        print(f"Wrote {len(images)} images and {len(annotations)} annotations to {out_json_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)