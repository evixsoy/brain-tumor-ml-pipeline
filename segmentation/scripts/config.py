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