import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import mixed_precision
import keras.backend as K
import albumentations as alb

from unet import create_unet_efficientnet

DATASET_FOLDER = 'dataset_split_segmentation'
MODEL_NAME = 'tumor_segmentation_best_512.keras'
CSV_LOG = 'training_512.csv'
MODEL = 'efficientnetb3'
INPUT_SIZE = (512, 512)
CLASSES = 2
CLASS_NAMES = ['background', 'tumor']

BATCH_SIZE = 32  
EPOCHS = 1000
MIXUP_PROB = 0.2
MIXUP_ALPHA = 0.2
INITIAL_LR = 5e-5 

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1 --tf_xla_cpu_global_jit=false'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

print("Using Stable Float32 (TF32) + Parallel Data Loading...")
mixed_precision.set_global_policy('float32')
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU Setup Error: {e}")

# augmentace
training_augmentation = alb.Compose([
    alb.HorizontalFlip(p=0.7),
    alb.VerticalFlip(p=0.3),
    alb.Rotate(limit=30, p=0.8),
    alb.Affine(scale=(0.8,1.2), translate_percent=(-0.15, 0.15), rotate=(-30, 30), shear=(-15, 15), p=0.7),
    alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    alb.RandomGamma(gamma_limit=(70, 130), p=0.6),
    alb.CLAHE(clip_limit=4.0, p=0.6),
    alb.CoarseDropout(num_holes_range=(2,6), hole_height_range=(10,20), hole_width_range=(10,20), p=0.3),
])

validation_augmentation = alb.Compose([])

#image generator
class LazySegmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_folder_path, mask_folder_path, batch_size, classes, 
                 input_size=INPUT_SIZE, augmentation=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.image_folder_path = image_folder_path
        self.mask_folder_path = mask_folder_path
        self.batch_size = batch_size
        self.classes = classes
        self.input_size = input_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        self.image_paths = [os.path.join(image_folder_path, f) for f in sorted(os.listdir(image_folder_path))]
        self.mask_paths = [os.path.join(mask_folder_path, f) for f in sorted(os.listdir(mask_folder_path))]
        
        self.indices = np.arange(len(self.image_paths))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_masks = []
        
        for i in batch_indices:
            image = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8)

            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)
            batch_images.append(image)
            batch_masks.append(mask)
        
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        batch_masks_onehot = to_categorical(batch_masks, num_classes=self.classes)

        if self.shuffle and np.random.random() < MIXUP_PROB:
            indices = np.random.permutation(len(batch_images))
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            batch_images = lam * batch_images + (1 - lam) * batch_images[indices]
            batch_masks_onehot = lam * batch_masks_onehot + (1 - lam) * batch_masks_onehot[indices]
        
        return batch_images, batch_masks_onehot
    
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# loss
def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    class_weights = [0.3, 0.7]
    total_tversky = 0.0
    for i in range(2):
        y_t = y_true[..., i]; y_p = y_pred[..., i]
        tp = tf.reduce_sum(y_t * y_p)
        fp = tf.reduce_sum((1 - y_t) * y_p)
        fn = tf.reduce_sum(y_t * (1 - y_p))
        tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        total_tversky += (1 - tversky_index) * class_weights[i]
    return total_tversky

def combined_loss(y_true, y_pred):
    return tversky_loss(y_true, y_pred)

#metrics
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
    dice_total = 0.0
    for i in range(2):
        y_t = tf.reshape(y_true[..., i], [-1]); y_p = tf.reshape(y_pred[..., i], [-1])
        intersection = tf.reduce_sum(y_t * y_p)
        dice_total += (2. * intersection + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)
    return dice_total / 2.0

def dice_coefficient_per_class(class_idx, class_name):
    def dice(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32); y_pred = tf.cast(y_pred, tf.float32)
        y_t = tf.reshape(y_true[..., class_idx], [-1]); y_p = tf.reshape(y_pred[..., class_idx], [-1])
        intersection = tf.reduce_sum(y_t * y_p)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)
    dice.__name__ = f'dice_{class_name}'
    return dice

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1); y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

metrics = [
    dice_coefficient,
    dice_coefficient_per_class(0, "background"),
    dice_coefficient_per_class(1, "tumor"),
    UpdatedMeanIoU(num_classes=CLASSES, name='mean_iou')
]

# setup generators
train_generator = LazySegmentationGenerator(f'{DATASET_FOLDER}/train/images', f'{DATASET_FOLDER}/train/masks', BATCH_SIZE, CLASSES, augmentation=training_augmentation)
val_generator = LazySegmentationGenerator(f'{DATASET_FOLDER}/val/images', f'{DATASET_FOLDER}/val/masks', BATCH_SIZE, CLASSES, shuffle=False)

#init
start_epoch = 0
best_val_dice = -np.inf 
custom_objects = {
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient,
    'dice_background': dice_coefficient_per_class(0, "background"),
    'dice_tumor': dice_coefficient_per_class(1, "tumor"),
    'UpdatedMeanIoU': UpdatedMeanIoU,
    'mean_iou': UpdatedMeanIoU(num_classes=CLASSES)
}

#check for existing .keras model
if os.path.exists('tumor_segmentation_last_512.keras'):
    print("Found latest epoch model. starting here")
    model_path = 'tumor_segmentation_last_512.keras'
elif os.path.exists('tumor_segmentation_best_512.keras'):
    print("Found best cloud model. starting here")
    model_path = 'tumor_segmentation_best_512.keras'
else:
    model_path = None

#resuming existing
if model_path:
    print(f"Loading weights from {model_path}")
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    if os.path.exists(CSV_LOG):
        df = pd.read_csv(CSV_LOG)
        if not df.empty:
            # Get last epoch
            start_epoch = int(df['epoch'].iloc[-1]) + 1
            # Get best val_dice_tumor to prevent overwriting better models
            if 'val_dice_tumor' in df.columns:
                best_val_dice = df['val_dice_tumor'].max()
            print(f"Resuming from epoch {start_epoch}, previous best Dice: {best_val_dice:.5f}")
else:
    model = create_unet_efficientnet(shape_input=(512, 512, 3), classes=CLASSES, decoder_sizes=[512, 256, 128, 64], dropout_rate=0.4, backbone='B3')

#new training
#warmup
if start_epoch < 5:
    print("warm up phase")
    model.compile(optimizer=Adam(1e-3), loss=combined_loss, metrics=metrics)
    model.fit(train_generator, 
              epochs=5, 
              initial_epoch=start_epoch, 
              validation_data=val_generator)
    start_epoch = 5

# Full training
for layer in model.layers: layer.trainable = True

model.compile(optimizer=Adam(learning_rate=INITIAL_LR, clipnorm=1.0), 
              loss=combined_loss, 
              metrics=metrics)

callbacks = [
    # save best model
    ModelCheckpoint('tumor_segmentation_best_512.keras', monitor='val_dice_tumor', verbose=1, save_best_only=True, mode='max', initial_value_threshold=best_val_dice if best_val_dice != -np.inf else None),
    # save latest
    ModelCheckpoint('tumor_segmentation_last_512.keras', verbose=0, save_best_only=False),
    EarlyStopping(monitor='val_dice_tumor', patience=100, verbose=1, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_dice_tumor', factor=0.5, patience=7, verbose=1, mode='max', min_lr=1e-6),
    CSVLogger(CSV_LOG, append=True),
]

print(f"Starting/Resuming training from epoch {start_epoch}...")
model.fit(train_generator, 
          epochs=EPOCHS, 
          initial_epoch=start_epoch, 
          validation_data=val_generator, 
          callbacks=callbacks)

model.save('tumor_segmentation_final_512_cloud.keras')
