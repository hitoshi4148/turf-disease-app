import os
import shutil
import random
from pathlib import Path

random.seed(42)

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_split"
SPLIT_RATIO = 0.8

classes = os.listdir(RAW_DIR)

for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_point]
    val_imgs = images[split_point:]

    for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(out_dir, exist_ok=True)

        for img in img_list:
            src = os.path.join(cls_path, img)
            dst = os.path.join(out_dir, img)
            shutil.copyfile(src, dst)

print("Dataset split completed.")