# Brain Tumor analyzing pipeline
- tensorflow/keras models for recognizing type and shape of tumor, growth prediction of tumor

## Including
- Classification model (glioma, pituitary, meningioma, no_tumor)
- Segmentation model (more about in description.md in directory)
- measuring script (takes tumor scan with scan spacing -> counts dimensions of  tumor)
- growth prediction (based on longitudinal data - not very reliable tbh)

### ~ Work in progress - currently working on frontend wrapper

## ToDo
 - Pipeline script
 - Moving from Tensorflow to PyTorch
 - Classification: using bigger dataset and change to double model (full-scan + tumor img only)
 - Segmentation: changing Architecture to prob. TransUnet (PyTorch), bigger dataset
 - growth prediction: using Segmentation for better prediction, more longitudinal data
 - possibly adding another features (3D visualization)
 - download_datasets.py script
 - cleaner, more modular code

###  ~ Models and datasets not yet included in repo (datasets are referenced in datasetref.txt )