## Segmentation model
512x512 deep learning model for recognizing tumors in brain mri scans

# Architecture

Attention U-net with EfficientNetB3 backbone (40M+ parameters)
	- decoder: [512, 256, 128, 64]
Tensorflow/keras 

## Data handling
~26k MRI brain scans across 5 datasets - all datasets except BraTS are in 512x512 (BraTS is in 240x240)
only

## Training
Data augmentations - using albumentations - CLAHE, CoarseDropout, Rotate...

Trained using rented A100 SXM4 80GB via [vast.ai](https://vast.ai/) - costed me 5 USD (0.678usd/hour) - highly recommended (using train_512_cloud.py) - trained for ~ 5-6hours -> 26 epochs (batch size 32)
>wanted to train on my rtx 3060 12gb - good for testing but wasted 100+ hours training (without a100 probably wouldn't have ever trained to these metrics) also have wanted to finetune the a100 model with my rtx 3060 but i would have probably wasted another 50+ hours for almost nothing



## Pipeline

Full Pipeline WIP 👀
>finished: 
>autodownload datasets 
>split datasets into validation, train, test, 
>save best and last versions of model training (.keras)



## Metrics

Validation_tumor_dice: 77.7%
Mean IoU: 88%
example visualization in repo
using dice + tvarsky loss

## In future

Aiming for 80-85% val_dice_tumor - more training with a100
make dataset bigger ofc
maybe go from unet to transunet or segformer (dont know much of them) - transfering to Pytorch
longitudinal dataset for growth prediction

## dataset links
https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification = scidb cast 0.49×0.49mm², slice thickness = 6mm, slice gap = 1mm (nevim co to znamena)
https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset - nemuzu najit
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation = 0.49 × 0.49 mm^2 slice thickness = 6mm, slice gap = 1mm (nevim co to znamena)
https://www.kaggle.com/datasets/briscdataset/brisc2025
https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

dataset1,dataset3 maji stejnou cast tech scanu -> casti o tom measuring jsou uplne stejne (oba datasety maji spolecnych asi DOST obrazku)
