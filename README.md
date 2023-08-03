# Modality-agnostic Augmented Multi-Collaboration Representation for Semi-supervised Heterogeneous Face Recognition

The zip file contains source codes we use in our paper for testing the Rank-1 accuracy on LAMP-HQ database.

## Dependencies

* Anaconda3 (Python 3.9, with Numpy etc.)
* Pytorch 1.12.1

## Datasets

[A Large-Scale Multi-Pose High-Quality Database (LAMP-HQ)](https://arxiv.org/abs/1912.07809) is a large-scale NIR-VIS face database with **56,788** NIR and **16,828** VIS images of **573** subjects. 

For each fold, the training set consists of almost 50% images from **300** subjects in the database. For the testing set, we select the rest **273** subjects, each with one VIS image and about **100** NIR images.


[CASIA NIR-VIS 2.0 dataset](https://ieeexplore.ieee.org/document/6595898) is widely used NIR-VIS face  dataset. This dataset contains **725** subjects. 

For each fold, the training set consists of about 6100 NIR images and **2500** VIS images from **360** identities. The test set consists of more than **6000** NIR and **358** VIS images from **358** identities, which are excluded from the training set.

[Tufts Face dataset](https://ieeexplore.ieee.org/document/8554155) is a large-scale public heterogeneous face database.

For the convenience of comparison, we choose the Thermal-VIS dataset, which contains **1583** paired thermal-VIS images from **112** subjects. we randomly select **90** subjects as the training set, the rest pairs are the testing set.

## Usage

### Prepare Database

1.Download face dataset such as CASIA NIR-VIS 2.0, LAMP-HQ and Tufts Face.

2.You can use RetinaFace for face detection and alignment, then crop face images to 128*128. For more information, please refer to: https://github.com/serengil/retinaface

### Test the model

1. run run_test.sh

