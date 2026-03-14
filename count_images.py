import os

root = "data_processed"

for d in sorted(os.listdir(root)):
    path = os.path.join(root, d)
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(f"{d}: {count}")