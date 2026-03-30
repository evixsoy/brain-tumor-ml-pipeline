import tensorflow as tf
from tensorflow import keras
from scripts.metrics_losses import combined_loss, dice_coefficient, dice_coefficient_per_class, UpdatedMeanIoU
from scripts.config import CLASSES, INPUT_SIZE
import cv2
import numpy as np
import os
import random

custom_objects = {
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient,
    'dice_background': dice_coefficient_per_class(0, "background"),
    'dice_tumor': dice_coefficient_per_class(1, "tumor"),
    'UpdatedMeanIoU': UpdatedMeanIoU,
    'mean_iou': UpdatedMeanIoU(num_classes=CLASSES)
}

def convert_to_rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        #grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 3:
        return img
    
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    print("invalid dimensions count")
    exit()

#TODO pridat check jestli uz je model loaded
#TODO pridat nejaky pokrocilejsi checky na obrazky
#TODO pridat lepsi pojmenovani
def predict(img_path:str, model_path:str = 'tumor_segmentation_best_512.keras'):
    if not os.path.exists(img_path):
        return "invalid path"
    
    img_orig = convert_to_rgb(cv2.imread(img_path))
    img = img_orig
    h_img, w_img = (img.shape[0], img.shape[1])

    #image check
    if w_img != h_img:
        return "image not square"
    
    if w_img < 64 :
        return "image too small"
    
    #resizing
    resized = False
    if (w_img,h_img) != INPUT_SIZE:
        resized = True
        if w_img > INPUT_SIZE[0]:
            #downscale
            img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        else:
            #upscale
            img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    img_resized = img

    #preprocess
    img_resized = np.expand_dims((img_resized.astype(np.float32) / 255.0), axis=0)
    
    #model pred
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Using model: {model_path}")
    prediction = model.predict(img_resized)[0]
    prediction_mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    mask_new = (prediction_mask * 255).astype(np.uint8)

    #1.create mask (from the 512x512 convert - resized)
    num = random.randint(0,1000)
    mask_path_new = f'predicted_masks/mask_{num}_new.png'
    cv2.imwrite(mask_path_new, mask_new)
    
    #2. create comparison image also from the corrected size
    contours, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    comparison_path_new = f'predicted_masks/comparison_{num}_new.png'
    cv2.imwrite(comparison_path_new, img)

    #1., 2. but for original img
    if resized:
        mask_orig = cv2.resize(mask_new, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_path_orig = f'predicted_masks/mask_{num}_orig.png'
        cv2.imwrite(mask_path_orig, mask_orig)
    
        #create comparison image - resized
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_orig, contours, -1, (0, 0, 255), 1)
        comparison_path_orig = f'predicted_masks/comparison_{num}_orig.png'
        cv2.imwrite(comparison_path_orig, img_orig)

    if resized:
        return [mask_path_new, comparison_path_new, mask_path_orig, comparison_path_orig]
    return [mask_path_new, comparison_path_new] 

# paths = predict("dummy_from_h5.png")
# paths = predict("dummy.jpg")
