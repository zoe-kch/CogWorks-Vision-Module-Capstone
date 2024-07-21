# Face Tracker
### Brought to you by We_Love_Bytes
![image](https://github.com/user-attachments/assets/effc65b3-6066-4bcd-8092-b9cadbfa9213)
<br/>
<br/>

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Contributing](#contributors)

## Introduction
> This project aims to perform facial recognition from a database of faces. It can be used in various implementations
> such as cameras for security and healthcare monitoring and hotel check ins/school attendance.

*Note: This project was created as part of BWSI CogWorks Week2 Vision Capstone project.*
*Presented by Team We Love Bytes 2024.*

## Features
These are some important features that our program can accomplish:
- [x] A fully trained Facenet model
- [x] Face-detection system
- [x] A database of recognized faces (includes Edwardia Fosah (one of this project's contributors), Davido, Lana Del Rey, Malcolm X, Issac Newton, Rema, Steve Lacy, and Taika Waititi initially)
- [x] Loading new faces to the database via dialog prompt
- [x] Recognizing known and unknown faces
- [x] Prototype** Whisper algorithm

## Installation

Before the installation of this program, we recommend you utilize a conda environment.
For more information about conda installation, visit https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

To set up the optimal conda environment, run the appropriate command and replace `env_name` with whatever you like.

For Windows/Linux, run:
```bash
conda create -n env_name python=3.8 jupyter notebook numpy matplotlib xarray numba bottleneck scipy opencv scikit-learn scikit-image pytorch torchvision cpuonly -c pytorch -c conda-forge
```

For Mac OS, run:
```bash
conda create -n env_name python=3.8 jupyter notebook numpy matplotlib xarray numba bottleneck scipy opencv scikit-learn scikit-image pytorch torchvision -c pytorch -c conda-forge
```

Activate the environment by running:
```bash
conda activate env_name
```

Once the environment is activate, run:
```bash
pip install mygrad mynn noggin facenet-pytorch cog-datasets
```

Next, install the [Camera](https://github.com/CogWorksBWSI/Camera/tree/master) and [facenet_models](https://github.com/CogWorksBWSI/facenet_models?tab=readme-ov-file) packages by following the instructions on the respective GitHubs. Note to make sure your Camera settings are properly configured and initialized prior to running main.py.

Navigate back to your original parent folder and clone this repository.
```bash
git clone https://github.com/zoe-kch/CogWorks-Vision-Module-Capstone.git
```
Navigate to our folder.
```bash
cd CogWorks-Vision-Module-Capstone
```
Install all dependencies.
```bash
pip install -r requirements.txt
```
And run main.py to start the program.
```bash
python main.py
```

## Usage
> It is important to use this technology with ethnics and consideration in mind. Privacy is a right.


## Configuration
We already optimized the configuration of this program. However,
you can still change some constant variables, such as threshold inside *find_thresholds.py*


## Contributors
- Zoe Granadoz
- Edwardia Fosah
- Bryan Wang
- Ye Yint Hmine
- Manya Tandon
