# Super Resolution
<p align="left">
    <a>
        <img src=https://img.shields.io/badge/python-3.6.12-green>
    </a>
    <a>
        <img src=https://img.shields.io/badge/pytorch-1.7.1-orange>
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

This repository gathers is the code for homework in class.
To read the detailed for what is this, please, refer to [my report](https://github.com/purpleFar/super-resolution/blob/master/readme_file/HW4_Report_0856735.pdf).

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.3 LTS
- Intel(R) Xeon(R) Platinum 8260M CPU @ 2.40GHz
- NVIDIA Tesla T4

## Outline
To reproduct my result without retrainig, do the following steps:
1. [Installation](#installation)
2. [Download Data](#download-data)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```bash=
$ conda create -n  hw1 python=3.6
$ conda activate drln_sr
$ pip install -r requirements.txt
```

## Download Data
The 291 training data download at [here](https://drive.google.com/drive/u/3/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x).
Unzip them then you can see following structure:
```
super-resolution/
    ├── model/
    ├── testing_lr_images
    │   ├── 00.png
    │   ├── 01.png
    │   │   .
    │   │   .
    │   │   .
    │   ├── 12.png
    │   └── 13.png
    ├── training_hr_images
    │   ├── 2092.png
    │   ├── 8049.png
    │   │   .
    │   │   .
    │   │   .
    │   └── tt27.png
    ├── data.py
    ├── make_submit.py
    ├── requirement.txt
    └── train.py
```

## Train models
To train models, run following command.
```bash=
$ mkdir result submit
$ python train.py
```

## Pretrained models
You can download pretrained model that used for my submission from [link](https://drive.google.com/file/d/1CGk4_meBMrPJuSDVh7Q03hTOWlrsXCAa/view?usp=sharing).
Unzip it and put into structure like below:
```
super-resolution/
    ├── model/
    ├── result
    │   └── best.pth
    ├── testing_lr_images/
    ├── training_hr_images/
    ├── data.py
    ├── make_submit.py
    ├── requirement.txt
    └── train.py
```

## Inference
If trained weights are prepared, you can create a folder containing the high resolution image for each picture in test set.

Using the pre-trained model, enter the command:
```bash=
$ python make_submit.py
```
or

If you want to train your model from scratch. Then the training code will also make the high-resolution image which is from `testing_lr_images` folder.

And you can see high-resolution images in folder which names `submit`.