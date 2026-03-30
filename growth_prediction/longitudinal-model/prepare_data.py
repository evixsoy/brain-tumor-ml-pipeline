import os
import shutil
import random
import json
import csv

#TODO filtrovat slozky ve kterych je mene jak tri scany (napriklad 3)

required_files = ["DeepBraTumIA-segmentation/atlas/skull_strip/ct1_skull_strip.nii.gz", "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz", "DeepBraTumIA-segmentation/atlas/segmentation/measured_volumes_in_mm3.json"]
useful_file = "DeepBraTumIA-segmentation/atlas/skull_strip/brain_mask.nii.gz"
dataset_path = os.path.join("dataset", "Imaging")

#stats
duplicate = 0
approved = 0
incomplete = 0

try:
    os.makedirs("filtered_dataset")
except Exception:
    shutil.rmtree("filtered_dataset")
    os.makedirs("filtered_dataset")

for patient_folder in sorted(os.listdir(dataset_path)):
    patient_path = os.path.join(dataset_path, patient_folder)
    for week_folder in sorted(os.listdir(patient_path)):
        week_path = os.path.join(patient_path, week_folder)
        
        not_complete = False
        for required in required_files:
            if not os.path.exists(os.path.join(week_path, required)):
                not_complete = True
                break
        if not_complete:
            incomplete +=1 
            continue

        is_useful_present = os.path.exists(os.path.join(week_path, useful_file))
        if len(week_folder) > 8:
            #duplicate scan
            week_folder = week_folder[:-2]
        week_filtered_path = os.path.join('filtered_dataset',patient_folder,week_folder)

        if not os.path.exists(week_filtered_path):
            os.makedirs(week_filtered_path, exist_ok = False)
            for required in required_files:
                shutil.copy(os.path.join(week_path, required), os.path.join(week_filtered_path, os.path.basename(required)))
            if is_useful_present:
                shutil.copy(os.path.join(week_path, useful_file), os.path.join(week_filtered_path, "brain_mask.nii.gz"))
            approved +=1 
        else:
            duplicate += 1
            continue

print(f"Stats: Approved: {approved} | Incomplete: {incomplete} | Duplicate: {duplicate}")

#make splits
random.seed(12)
ids = []
for patient_folder in os.listdir('filtered_dataset'):
    ids.append(patient_folder[-3:])

random.shuffle(ids)
l = len(ids)
train_ids = ids[:int(0.7*l)]
test_ids = ids[int(0.7*l):int(0.85*l)]
val_ids = ids[int(0.85*l):]
splits = {
    "train_ids": sorted(train_ids),
    "test_ids": sorted(test_ids),
    "val_ids": sorted(val_ids)
}

with open("splits.json", "w") as f:
    json.dump(splits, f,)

#generate pairs.csv
data = [["patient_id", "week_t", "week_t1", "week_t_path", "week_t1_path", "delta_weeks", "y_growth_per_week_mm3" ]]
for patient_folder in sorted(os.listdir('filtered_dataset')):
    patient_id = patient_folder[-3:]
    patient_path = os.path.join("filtered_dataset", patient_folder)
    weeks = sorted(os.listdir(patient_path))
    week_pairs = [(weeks[i], weeks[i+1]) for i in range(len(weeks)-1)]

    for week_t, week_t1 in week_pairs:
        week_t_path = os.path.join(patient_path, week_t)
        with open(os.path.join(week_t_path, "measured_volumes_in_mm3.json"), "r") as f:
            week_t_json_data = json.load(f)
        week_t_enchancing_core = week_t_json_data["Enhancing_Core"]

        week_t1_path = os.path.join(patient_path, week_t1)
        with open(os.path.join(week_t1_path, "measured_volumes_in_mm3.json"), "r") as f:
            week_t1_json_data = json.load(f)
        week_t1_enchancing_core = week_t1_json_data["Enhancing_Core"]

        delta_weeks = int(week_t1[-3:]) - int(week_t[-3:])
        growth_per_week_mm3 = (week_t1_enchancing_core - week_t_enchancing_core) / delta_weeks
        data.append([patient_id, week_t, week_t1, week_t_path, week_t1_path, delta_weeks, growth_per_week_mm3])

with open("pairs.csv", "w")as f:
    writer = csv.writer(f)
    writer.writerows(data)