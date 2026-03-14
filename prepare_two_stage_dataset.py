import os
import shutil


RAW_DIR = "data_raw"
OUT_ROOT = "data_two_stage"
STAGE1_DIR = os.path.join(OUT_ROOT, "stage1_binary")
STAGE2_DIR = os.path.join(OUT_ROOT, "stage2_disease")

HEALTHY_CLASS = "healthy"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def list_classes(root):
    return sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    )


def copy_class_images(src_dir, dst_dir, prefix=None):
    os.makedirs(dst_dir, exist_ok=True)
    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        if not name.lower().endswith(IMAGE_EXTS):
            continue

        # stage1/disease で同名衝突を避ける
        out_name = f"{prefix}__{name}" if prefix else name
        shutil.copy2(src, os.path.join(dst_dir, out_name))


def main():
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"{RAW_DIR} が見つかりません。")

    classes = list_classes(RAW_DIR)
    disease_classes = [c for c in classes if c != HEALTHY_CLASS]

    ensure_clean_dir(STAGE1_DIR)
    ensure_clean_dir(STAGE2_DIR)

    stage1_healthy_dir = os.path.join(STAGE1_DIR, "healthy")
    stage1_disease_dir = os.path.join(STAGE1_DIR, "disease")
    os.makedirs(stage1_healthy_dir, exist_ok=True)
    os.makedirs(stage1_disease_dir, exist_ok=True)

    # 1) healthy -> stage1_binary/healthy
    healthy_src = os.path.join(RAW_DIR, HEALTHY_CLASS)
    if os.path.exists(healthy_src):
        copy_class_images(healthy_src, stage1_healthy_dir)

    # 2) disease classes -> stage1_binary/disease (統合)
    for cls in disease_classes:
        src = os.path.join(RAW_DIR, cls)
        copy_class_images(src, stage1_disease_dir, prefix=cls)

    # 3) disease classes -> stage2_disease/<class>
    for cls in disease_classes:
        src = os.path.join(RAW_DIR, cls)
        dst = os.path.join(STAGE2_DIR, cls)
        copy_class_images(src, dst)

    print("Done: two-stage dataset prepared")
    print("Stage1:", STAGE1_DIR)
    print("Stage2:", STAGE2_DIR)


if __name__ == "__main__":
    main()
