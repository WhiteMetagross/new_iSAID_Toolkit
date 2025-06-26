import os
import json
import shutil
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert preprocessed iSAID dataset to YOLO segmentation format.")
    parser.add_argument('--datadir', type=str, default='./iSAID_patches',
                        help="Path to the root directory of the preprocessed iSAID dataset (input).")
    parser.add_argument('--outdir', type=str, default='./iSAID_YOLO_Dataset',
                        help="Path to the root directory where the YOLO formatted dataset will be saved (output).")
    return parser.parse_args()

def convert_isaid_to_yolo_seg(input_root: str, output_root: str):
    print(f"Starting conversion from '{input_root}' to YOLO format at '{output_root}'...")
    
    for split in ("train", "val", "test"):
        (Path(output_root) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(output_root) / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        json_path = Path(input_root) / split / f"instancesonly_filtered_{split}.json"
        if not json_path.exists():
            print(f"Warning: JSON file not found for '{split}' split. Skipping: {json_path}")
            continue
        
        print(f"Processing '{split}' split...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        images = {img["id"]: img for img in data["images"]}
        
        annos = {}
        for ann in data["annotations"]:
            annos.setdefault(ann["image_id"], []).append(ann)
            
        cat_ids = sorted(c["id"] for c in data["categories"])
        catid2idx = {cid: idx for idx, cid in enumerate(cat_ids)}

        for img_id, img_info in images.items():
            fname = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]
            
            src_img_path = Path(input_root) / split / "images" / fname
            dst_img_path = Path(output_root) / "images" / split / fname
            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Warning: Source image not found: {src_img_path}")
            
            label_path = Path(output_root) / "labels" / split / f"{Path(fname).stem}.txt"
            with open(label_path, "w") as f:
                for ann in annos.get(img_id, []):
                    cls_idx = catid2idx[ann["category_id"]]
                    
                    if ann.get("segmentation") and len(ann["segmentation"]) > 0:
                        seg = ann["segmentation"][0]
                        if len(seg) >= 6 and len(seg) % 2 == 0:
                            seg_n = [coord / w if i % 2 == 0 else coord / h for i, coord in enumerate(seg)]
                            
                            parts = [str(cls_idx)]
                            parts += [f"{v:.6f}" for v in seg_n]
                            f.write(" ".join(parts) + "\n")

    print("Processing 'test' split...")
    test_json_path = Path(input_root) / "test" / "instancesonly_filtered_test.json"
    if test_json_path.exists():
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
        for img in test_data["images"]:
            fname = img["file_name"]
            src_img_path = Path(input_root) / "test" / "images" / fname
            dst_img_path = Path(output_root) / "images" / "test" / fname
            
            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"Warning: Source image not found: {src_img_path}")
                
            label_path = Path(output_root) / "labels" / "test" / f"{Path(fname).stem}.txt"
            with open(label_path, "w") as f:
                pass

    print("Writing data.yaml file...")
    train_json_path = Path(input_root) / "train" / "instancesonly_filtered_train.json"
    if train_json_path.exists():
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        names = [c["name"] for c in sorted(train_data["categories"], key=lambda x: x["id"])]
        
        yaml_path = Path(output_root) / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"nc: {len(names)}\n")
            f.write(f"names: {names}\n")
    else:
        print("Warning: Could not find train JSON file to extract class names for data.yaml")
    
    print("Conversion complete.")

if __name__ == "__main__":
    args = parse_args()
    convert_isaid_to_yolo_seg(args.datadir, args.outdir)