import tensorflow as tf
import joblib
import numpy as np
import SimpleITK as sitk
import json
from tensorflow import keras

# prediction_in_time -> 1 week, 20 weeks etc
#tumor_volume_mm3 = Enhancing_Core

def predict(img_path:str, mask_path:str, lag_features_json_path:str, tumor_volume_mm3: float | int, necrotic_nonenh: float | int = 0, edema_compartment: float | int = 0, prediction_time: int = 5, model_path:str = 'model/model.keras', scaler_path:str = 'model/scaler.pkl'):
    #feature extraction for week_t scan (latest scan)   
    week_t_ct1 = sitk.ReadImage(img_path)
    week_t_image = sitk.GetArrayFromImage(week_t_ct1)
    week_t_mask = sitk.ReadImage(mask_path)
    week_t_mask = sitk.GetArrayFromImage(week_t_mask)
    week_t_mask_binary = (week_t_mask > 0)

    #features
    #bbox volume
    coords = np.argwhere(week_t_mask_binary > 0) # -> (z,y,x)
    z_min,y_min,x_min = coords.min(axis=0)
    z_max,y_max,x_max = coords.max(axis=0)
    bbox_voxels = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
    
    sx,sy,sz = week_t_ct1.GetSpacing()
    voxel_volume = sx * sy * sz # 1 voxel to mm3
        
    bbox_volume_mm3 = bbox_voxels * voxel_volume

    #extent
    extent = tumor_volume_mm3 / bbox_volume_mm3
    
    # - mean intensity in mask
    vals = week_t_image[week_t_mask_binary]
    mean_intensity = vals.mean()

    # - std intensity in mask
    std_intensity = vals.std()

    p25_intensity = np.percentile(vals,25)

    p75_intensity = np.percentile(vals,75)

    voxel_mask_count = week_t_mask_binary.sum()

    delta_weeks = prediction_time #model returns per week growth! 
    
    with open(lag_features_json_path, "r") as f:
        lag_features = json.load(f)
    
    lag_y1 = lag_features["lag_y1"]
    lag_y2 = lag_features["lag_y2"]
    has_lag1 = eval(lag_features["has_lag1"])
    has_lag2 = eval(lag_features["has_lag2"])

    x = [tumor_volume_mm3, bbox_volume_mm3, voxel_mask_count, extent, mean_intensity, std_intensity, p25_intensity, p75_intensity, necrotic_nonenh, edema_compartment, delta_weeks, lag_y1, has_lag1, lag_y2, has_lag2]

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    x = np.array([x], dtype=np.float32)
    x = scaler.transform(x)

    pred_per_week = model.predict(x, verbose=0).ravel()[0]
    pred_total = pred_per_week * prediction_time

    delta_growth = (((pred_total + tumor_volume_mm3 ) / tumor_volume_mm3) *100) - 100
    if delta_growth > 15:
        label = "Growing"
    elif delta_growth < 15 and delta_growth > -15:
        label = "Stable"
    else:
        label = "Shrinking"

    return tuple((label, f"{float(delta_growth):.2f}% tumor growth", f"{float(pred_per_week):.2f}mm^2 per week", f"{float(pred_total):.2f}mm^2 total"))

# print(predict("ct1_skull_strip.nii.gz","seg_mask.nii.gz","test.json", 709, 3354, 32472, 11 ))