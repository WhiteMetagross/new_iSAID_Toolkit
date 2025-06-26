import argparse
import json
import os
import cv2
from natsort import natsorted

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
    sets = args.set.split(',')
    
    for data_set in sets:
        if data_set != 'test':
            print(f"Skipping non-test set: {data_set}")
            continue

        ann_dir = os.path.join(args.datadir, data_set, 'images')
        print(f"Scanning {ann_dir} for test images...")

        if not os.path.exists(ann_dir):
            print(f"Directory not found: {ann_dir}")
            continue

        images = []
        img_id = 0
        for root, _, files in os.walk(ann_dir):
            for filename in natsorted(files):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if '_instance_color_RGB' in filename or '_instance_id_RGB' in filename:
                        continue
                        
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: failed to read {filename}")
                        continue
                    h, w = img.shape[:2]
                    
                    images.append({
                        'id': img_id,
                        'width': w,
                        'height': h,
                        'file_name': filename
                    })
                    img_id += 1

        ann_dict = {
            'images': images,
            'categories': get_category_info(),
            'annotations': []
        }
        
        os.makedirs(os.path.join(args.outdir, data_set), exist_ok=True)
        out_file = os.path.join(args.outdir, data_set, 'instancesonly_filtered_test.json')
        with open(out_file, 'w') as f:
            json.dump(ann_dict, f, indent=4)
        print(f"Wrote {len(images)} test image entries to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test image IDs JSON')
    parser.add_argument('--outdir', default='./iSAID_patches', type=str)
    parser.add_argument('--datadir', default='./iSAID_patches', type=str)
    parser.add_argument('--set', default="test", type=str)
    args = parser.parse_args()
    main(args)