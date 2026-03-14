import os
import cv2
import random
from tqdm import tqdm
import albumentations as A

# ========= 設定 =========
RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
TARGET_COUNT = 150
IMG_SIZE = 224

# ========= Augmentation =========
transform = A.Compose([
    A.Rotate(limit=10, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.1,
        p=0.7
    ),
    A.ShiftScaleRotate(
        shift_limit=0.02,
        scale_limit=0.05,
        rotate_limit=10,
        p=0.5
    ),
    A.Resize(IMG_SIZE, IMG_SIZE)
])

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    output_class_path = os.path.join(OUTPUT_DIR, class_name)

    ensure_dir(output_class_path)

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"{class_name}: {len(images)} files")
    if not images:
        return

    # 元画像コピー（リサイズのみ）
    copied_count = 0
    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ Skipping unreadable file: {img_name}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(output_class_path, img_name), img)
        copied_count += 1

    current_count = copied_count

    if current_count >= TARGET_COUNT:
        return

    needed = TARGET_COUNT - current_count
    print(f"  → Augmenting {needed} images")

    for i in tqdm(range(needed)):
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = transform(image=img)
        aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)

        new_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(output_class_path, new_name), aug_img)


def main():
    ensure_dir(OUTPUT_DIR)

    classes = os.listdir(RAW_DIR)

    for cls in classes:
        process_class(cls)

    print("Done!")

if __name__ == "__main__":
    main()