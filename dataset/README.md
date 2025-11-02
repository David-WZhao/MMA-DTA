# MMA-DTA

This is the code for 《Molecular Mechanics-aware Feature Fusion Framework for Predicting Protein-ligand Binding Affinity》. 

## Environment

The following dependencies are required to run the code:

- Python 3.7+
- numpy
- pandas
- scipy
- torch
- torch-geometric
- rdkit
- Biopython

## Dateset

The dataset is available at https://www.pdbbind-plus.org.cn/.

## Running feature_capture_save_features_5_19

This script builds and trains a deep learning model using the captured molecular features. To run the model training process, execute:

```bash
python feature_capture_save_features_5_19.py
```

This will process the structures and save features in the specified output directory.

## Running Fusion_5_25.py

```
python Fusion_5_25.py
```

Make sure to specify the correct paths to your input data, including the molecular features and affinity data.