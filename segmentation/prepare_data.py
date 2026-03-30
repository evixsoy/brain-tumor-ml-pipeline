import os
import shutil
import hashlib
import random
import csv
import cv2
import numpy as np
import h5py


os.makedirs('grouped_dataset/images', exist_ok =True)
os.makedirs('grouped_dataset/masks', exist_ok =True)

# Dataset 1
dataset_path = "datasets/dataset1"
for tumor_folder in os.listdir(os.path.join(dataset_path, "image")):
    if not os.path.isdir(os.path.join(dataset_path, "image", tumor_folder)): continue
    for file in os.listdir(os.path.join(dataset_path, "image", tumor_folder)):
        image_filename = os.path.join(dataset_path, "image", tumor_folder, file)
        mask_filename = os.path.join(dataset_path, "mask", tumor_folder, file).replace('.jpg', '_m.jpg')
        if os.path.exists(mask_filename):
            shutil.copy(image_filename, os.path.join("grouped_dataset/images", file))
            shutil.copy(mask_filename, os.path.join("grouped_dataset/masks", file))

# Dataset 2
dataset_path = "datasets/dataset2"
for file in os.listdir(os.path.join(dataset_path, "images")):
    image_path = os.path.join(dataset_path, "images", file)
    mask_path = os.path.join(dataset_path, "masks", file)
    if os.path.exists(mask_path):
        shutil.copy(image_path, os.path.join("grouped_dataset/images", file))
        shutil.copy(mask_path, os.path.join("grouped_dataset/masks", file))

# Dataset 3
dataset_path = "datasets/dataset3/segmentation_task"
for type_folder in os.listdir(dataset_path):
    if not os.path.isdir(os.path.join(dataset_path, type_folder)): continue
    for file in os.listdir(os.path.join(dataset_path, type_folder, "images")):
        image_path = os.path.join(dataset_path, type_folder, "images", file)
        mask_path = os.path.join(dataset_path, type_folder, "masks", file).replace('.jpg', '.png')
        if os.path.exists(mask_path):
            shutil.copy(image_path, os.path.join("grouped_dataset/images", file))
            shutil.copy(mask_path, os.path.join("grouped_dataset/masks", file))

# Dataset 4
dataset_path = "datasets/dataset4/Segmentation-masks&images"
for type_folder in os.listdir(dataset_path):
    if not os.path.isdir(os.path.join(dataset_path, type_folder)): continue
    for file in os.listdir(os.path.join(dataset_path, type_folder)):
        if file[-5] == "k": continue
        image_path = os.path.join(dataset_path, type_folder, file)
        mask_path = image_path[:-4] + "_mask.png"
        if os.path.exists(mask_path):
            shutil.copy(image_path, os.path.join("grouped_dataset/images", file))
            shutil.copy(mask_path, os.path.join("grouped_dataset/masks", file))


# brats dataset5
csv_path = "datasets/dataset5/BraTS20 Training Metadata.csv"
with open(csv_path, 'r') as f:
    data = list(csv.DictReader(f))

# Filtrace
target_files = [row for row in data if row['target'] == '1' and int(row['label1_pxl_cnt']) > 150]
no_tumor_files = [row for row in data if row['target'] == '0' and 45 < int(row['slice']) < 120]
if len(no_tumor_files) > 5000: no_tumor_files = random.sample(no_tumor_files, 5000)
dataset_files = target_files + no_tumor_files
random.shuffle(dataset_files)

temp_path = 'dataset5_temp'
if os.path.exists(temp_path): shutil.rmtree(temp_path)
os.makedirs(f'{temp_path}/images', exist_ok=True)
os.makedirs(f'{temp_path}/masks', exist_ok=True)

#ai code for convert brats h5 to png - needs rework ! 
def crop_brain(img, mask, size=240):
    coords = np.argwhere(img > 0)
    if coords.size == 0: return img, mask
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
    h = size // 2
    y1, x1 = max(0, cy - h), max(0, cx - h)
    y2, x2 = min(img.shape[0], y1 + size), min(img.shape[1], x1 + size)
    if y2 == img.shape[0]: y1 = max(0, y2 - size)
    if x2 == img.shape[1]: x1 = max(0, x2 - size)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

for row in dataset_files:
    h5_path_temp = row['slice_path'].replace('../input/brats2020-training-data/', '')
    h5_path = os.path.join(os.getcwd(), 'datasets/dataset5', h5_path_temp)
    if not os.path.exists(h5_path): continue
    try:
        with h5py.File(h5_path, 'r') as f:
            img_data = f['image'][:]; mask_data = f['mask'][:]
            mask_combined = np.max(mask_data, axis=-1)
            flair = img_data[:, :, 3]
            p1, p99 = np.percentile(flair, [1, 99])
            img_norm = np.clip((flair-p1)/(p99-p1), 0, 1)*255 if p99>p1 else flair*0
            img_c, mask_c = crop_brain(img_norm, mask_combined, size=240)
            base = f"brats_{os.path.basename(h5_path).replace('.h5', '')}.png"
            cv2.imwrite(os.path.join(temp_path, "images", base), img_c.astype(np.uint8))
            cv2.imwrite(os.path.join(temp_path, "masks", base), (mask_c * 255).astype(np.uint8))
    except Exception as e : 
        print(f"Error processing {h5_path}: {e}")

# sjednoceni jmen
def rename_and_copy(path, folder_type):
    files = sorted(os.listdir(path))
    for num, file in enumerate(files):
        id = str(num).zfill(4)
        curr_filepath = os.path.join('grouped_dataset', folder_type, file)
        new_filepath = os.path.join('grouped_dataset', folder_type, f"{id}_tumor.jpg")
        os.rename(curr_filepath, new_filepath)

# check na duplikaty
hashes = set()
for file in sorted(os.listdir('grouped_dataset/images')):
    image_path = os.path.join("grouped_dataset/images", file)
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash not in hashes: hashes.add(file_hash)
        else:
            os.remove(image_path)
            os.remove(f"grouped_dataset/masks/{file}")

# split
random.seed(12)
images_path = 'grouped_dataset/images/'
files = os.listdir(images_path)
random.shuffle(files)
l = len(files)
train = files[:int(0.7*l)]; test = files[int(0.7*l):int(0.85*l)]; val = files[int(0.85*l):]

for split, f_list in zip(["train", "test", "val"], [train, test, val]):
    os.makedirs(f"dataset_split_segmentation/{split}/images/", exist_ok=True)
    os.makedirs(f"dataset_split_segmentation/{split}/masks/", exist_ok=True)
    for filename in f_list:
        shutil.move(os.path.join("grouped_dataset", "images", filename), f"dataset_split_segmentation/{split}/images/{filename}")
        shutil.move(os.path.join("grouped_dataset", "masks", filename), f"dataset_split_segmentation/{split}/masks/{filename}")

if os.path.exists("grouped_dataset"): shutil.rmtree("grouped_dataset")

# brats volume-wise split
unique_volumes = sorted(list(set(int(row['volume']) for row in dataset_files)))
random.shuffle(unique_volumes)
n = len(unique_volumes)
train_ids = set(unique_volumes[:int(0.7*n)])
test_ids = set(unique_volumes[int(0.7*n):int(0.85*n)])
val_ids = set(unique_volumes[int(0.85*n):])

for filename in os.listdir(os.path.join(temp_path, 'images')):
    text_parts = filename.split("_")
    try:
        vol_id = int(text_parts[2])
    except (IndexError, ValueError):
        continue

    if vol_id in train_ids:
        split_folder = "train"
    elif vol_id in val_ids:
        split_folder = "val"
    else:
        split_folder = "test"
    
    shutil.move(os.path.join(temp_path, 'images', filename), os.path.join("dataset_split_segmentation", split_folder, "images", filename))
    shutil.move(os.path.join(temp_path, 'masks', filename), os.path.join("dataset_split_segmentation", split_folder, "masks", filename))

if os.path.exists(temp_path): shutil.rmtree(temp_path)