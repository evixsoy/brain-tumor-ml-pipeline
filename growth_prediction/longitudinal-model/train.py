import os 
import json
import pandas as pd
import numpy as np
import SimpleITK as sitk
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import random

SEED = 12
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

pairs = pd.read_csv("pairs.csv", dtype={"patient_id": str}) 
pairs = pairs.sort_values(["patient_id", "week_t"])
patient_ids = list(set(pairs["patient_id"]))

#get lag feature(t-1, t-2 info) - TODO presunout do prepare_data.py
patient_groups = pairs.groupby("patient_id")
pairs["lag_y1"] = patient_groups["y_growth_per_week_mm3"].shift(1)
pairs["lag_y2"] = patient_groups["y_growth_per_week_mm3"].shift(2)
pairs["lag_y1"] = pairs["lag_y1"].fillna(0.0)
pairs["lag_y2"] = pairs["lag_y2"].fillna(0.0)
pairs["has_lag1"] = (pairs["lag_y1"] != 0).astype(float)
pairs["has_lag2"] = (pairs["lag_y2"] != 0).astype(float)


#create splits
with open("splits.json", "r") as f:
    splits = json.load(f)

train_ids = splits["train_ids"]
x_train, y_train = [], []
test_ids = splits["test_ids"]
x_test, y_test = [], []
val_ids = splits["val_ids"]
x_val, y_val = [], []

invalid_rows = []
meta_test_rows = []

for patient_id in patient_ids:
    weeks_t_paths = list(pairs.loc[pairs["patient_id"] == patient_id, "week_t_path"])
    for week_t_path in weeks_t_paths:
        ct1_path = os.path.join(week_t_path, "ct1_skull_strip.nii.gz")
        mask_path = os.path.join(week_t_path, "seg_mask.nii.gz")
        
        ct1 = sitk.ReadImage(ct1_path)
        mask = sitk.ReadImage(mask_path)

        if ct1.GetSize() != mask.GetSize() or ct1.GetDimension() != mask.GetDimension():
            print("invalid")
            continue
        
        image = sitk.GetArrayFromImage(ct1)
        mask = sitk.GetArrayFromImage(mask)
        mask_binary = (mask > 0)

        if mask_binary.sum() == 0:
            invalid_rows.append((patient_id, week_t_path))
            continue
        
        # 2. feature extraction - required
        # - tumor volume in mm3
        with open(os.path.join(week_t_path, "measured_volumes_in_mm3.json")) as f:
            json_data = json.load(f)
        tumor_volume_mm3 = json_data["Enhancing_Core"]

        # - bbox volume feature
        coords = np.argwhere(mask_binary > 0) # -> (z,y,x)
        z_min,y_min,x_min = coords.min(axis=0)
        z_max,y_max,x_max = coords.max(axis=0)
        bbox_voxels = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
        
        sx,sy,sz = ct1.GetSpacing()
        voxel_volume = sx * sy * sz # 1 voxel to mm3
        
        bbox_volume_mm3 = bbox_voxels * voxel_volume
        
        # - extent
        extent = tumor_volume_mm3 / bbox_volume_mm3

        # - mean intensity in mask
        vals = image[mask_binary]
        mean_intensity = vals.mean()

        # - std intensity in mask
        std_intensity = vals.std()

        # - p25 intensity
        p25_intensity = np.percentile(vals,25)
        # - p75 intensity
        p75_intensity = np.percentile(vals,75)

        # - voxels in mask
        voxel_mask_count = mask_binary.sum()

        curr_row = pairs.loc[(pairs["patient_id"] == patient_id) & (pairs["week_t_path"] == week_t_path)]
        # - delta weeks
        delta_weeks = curr_row["delta_weeks"].iloc[0]
        
        # - lag feature (t-1, t-2 info)
        lag_y1 = curr_row["lag_y1"].iloc[0]
        has_lag1 = curr_row["has_lag1"].iloc[0]
        lag_y2 = curr_row["lag_y2"].iloc[0]
        has_lag2 = curr_row["has_lag2"].iloc[0]

        #history : 889 - lag1 a haslag1, lag2 a haslag2 batch size 16
        # 2. feature extraction - optional
        necrotic_nonenh = json_data["Necrotic_NonEnhancing"]
        edema_compartment = json_data["Edema_Compartment"]

        y = curr_row["y_growth_per_week_mm3"].iloc[0]

        x = [tumor_volume_mm3, bbox_volume_mm3,voxel_mask_count, extent, mean_intensity, std_intensity, p25_intensity, p75_intensity, necrotic_nonenh ,edema_compartment, delta_weeks, lag_y1, has_lag1, lag_y2, has_lag2]

        if voxel_mask_count < 1 or curr_row.empty or not np.all(np.isfinite(x)):
           invalid_rows.append((patient_id, week_t_path))
           continue

        if patient_id in train_ids:
            x_train.append(x)
            y_train.append(y)
        elif patient_id in val_ids:
            x_val.append(x)
            y_val.append(y)
        else:
            x_test.append(x)
            y_test.append(y)
            meta_test_rows.append({
                "patient_id": patient_id,
                "week_t": curr_row["week_t"].iloc[0],
                "week_t1": curr_row["week_t1"].iloc[0],
            })

X_train = np.array(x_train, dtype=np.float32)
Y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(x_val, dtype=np.float32)
Y_val = np.array(y_val, dtype=np.float32)
X_test = np.array(x_test, dtype=np.float32)
Y_test = np.array(y_test, dtype=np.float32)

#normalization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

#baseline test
Y_pred_baseline = np.zeros_like(Y_test)
baseline_mae = np.mean(np.abs(Y_test - Y_pred_baseline))
print("baseline MAE:", baseline_mae)

#model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(64, activation="relu"),
    Dense(1)
  ])
model.compile(optimizer=Adam(1e-3), loss="mae", metrics=["mae"])

callbacks = [EarlyStopping(monitor="val_mae", mode="min", patience=10, min_delta=5.0, restore_best_weights=True),
            ModelCheckpoint("checkpoint.keras", save_best_only=True)]

model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
        epochs=200, batch_size=16, callbacks=callbacks)

Y_pred = model.predict(X_test, verbose= 0).ravel()
mae = np.mean(np.abs(Y_test - Y_pred))

print("pred:", mae)

os.makedirs("model", exist_ok=True)
model.save(f"model/model.keras")
joblib.dump(scaler, f"model/scaler.pkl")

#TODO pridat vytvareni predictions.csv jak v history?
