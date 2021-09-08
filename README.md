# HRCN
This repository contains PyTorch codes for the ICCV2021 paper "**Heterogeneous Relational Complement for Vehicle Re-identification**"

## Installation
### Requirements
* Linux with python 3.6
* pytorch 1.4.0  
* torchvision 0.5.0
* cudatoolkit 10.0

### Set up with Conda
```
cd HRCN
conda env create -f hrcn.yml
conda activate hrcn
pip install -r requirements.txt
```

## Training and Evaluating
Replace the [source_link] with the dataset directory in *dataset_soft_link.sh*.

Download [trained models](https://drive.google.com/drive/folders/1gDz761-gTF3nLnwU24kDIVzDbCyBJu80?usp=sharing) into the directory *model_weight*. 

```
cd HRCN
sh dataset_soft_link.sh

# Train in VehicleID, VeRi or VERIWild
sh trainVehicleID.sh
sh trainVeRi.sh
sh trainVERIWild.sh

# Evaluate in VehicleID, VeRi or VERIWild
sh testVehicleID.sh
sh testVeRi.sh
sh testVERIWild.sh
```

## Citation
To be updated.

## Acknowledgment
This repository is based on the implementation of [fast-reid](https://github.com/JDAI-CV/fast-reid).
