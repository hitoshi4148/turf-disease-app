from PIL import Image
import os

root = "data_processed"
bad_files = []

for root_dir, dirs, files in os.walk(root):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root_dir, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                bad_files.append(path)

print("Bad files:", len(bad_files))

for f in bad_files:
    print(f)