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
```
@InProceedings{Zhao_2021_ICCV,
    author    = {Zhao, Jiajian and Zhao, Yifan and Li, Jia and Yan, Ke and Tian, Yonghong},
    title     = {Heterogeneous Relational Complement for Vehicle Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {205-214}
}
```
## Acknowledgment
This repository is based on the implementation of [fast-reid](https://github.com/JDAI-CV/fast-reid).
