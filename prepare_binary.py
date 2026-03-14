import os
import shutil

SOURCE_DIR = "data_processed"
TARGET_DIR = "data_binary"

HEALTHY_CLASS = "healthy"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    ensure_dir(TARGET_DIR)

    healthy_dir = os.path.join(TARGET_DIR, "healthy")
    disease_dir = os.path.join(TARGET_DIR, "disease")

    ensure_dir(healthy_dir)
    ensure_dir(disease_dir)

    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(class_path)

        for img in images:
            src = os.path.join(class_path, img)

            if cls == HEALTHY_CLASS:
                dst = os.path.join(healthy_dir, f"{cls}_{img}")
            else:
                dst = os.path.join(disease_dir, f"{cls}_{img}")

            shutil.copy(src, dst)

    print("Binary dataset prepared!")

if __name__ == "__main__":
    main()