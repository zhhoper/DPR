<!--<h3><b>DPR</b></h3>-->
## <b>Deep Single-Image Portrait Relighting</b> [[Project Page]](http://zhhoper.github.io/dpr.html) <br>
Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, David W. Jacobs. In ICCV, 2019

<p><img src="result/obama_00.jpg" width="120px" >
<img src="result/obama_01.jpg" width="120px" >
<img src="result/obama_02.jpg" width="120x" >
<img src="result/obama_03.jpg" width="120px" >
<img src="result/obama_04.jpg" width="120px" >
<img src="result/obama_05.jpg" width="120px" >
<img src="result/obama_06.jpg" width="120px" >
</p>
<p><img src="result/light_00.png" width="120px" >
<img src="result/light_01.png" width="120px" >
<img src="result/light_02.png" width="120x" >
<img src="result/light_03.png" width="120px" >
<img src="result/light_04.png" width="120px" >
<img src="result/light_05.png" width="120px" >
<img src="result/light_06.png" width="120px" >
</p>

---
# Table of Contents

- [Deep Single-Image Portrait Relighting](#deep-single-image-portrait-relighting)
  - [Overview](#overview)
  - [Notes](#notes)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
    - [Setup Instructions](#setup-instructions)
  - [Brightness Configuration File](#brightness-configuration-file)
  - [Running the Project](#running-the-project)
  - [Data Preparation](#data-preparation)
  - [Citation](#citation)
---
# Overview
This project is an implementation of the paper "Deep Single-Image Portrait Relighting" presented at ICCV 2019. Its core goal is to perform relighting on single portrait images using deep learning: given an input portrait and a target lighting configuration, the model can re-render the same face under completely new illumination conditions. The method is fundamentally based on intrinsic image decomposition, where the network learns to separate an input image into its physically meaningful components—albedo (surface reflectance), geometry (represented by a depth map or surface normal vectors), and scene lighting. By disentangling these factors and then replacing the estimated lighting component with a desired target profile, the system renders the final output through a data-driven pipeline rooted in classic image formation models, enabling precise and controllable illumination effects.

From a technical standpoint, the architecture relies on a U-Net with an Hourglass structure that preserves spatial resolution while extracting features. A key design choice is the use of second-order Spherical Harmonics (SH) for lighting representation, employing 9 coefficients: one for ambient light, three for directional lighting along the coordinate axes, and five for higher-frequency details such as soft shadows on curved facial surfaces. This compact yet physically grounded representation makes illumination manipulation both efficient and interpretable. The codebase supports processing images at 512×512 and 1024×1024 resolutions and depends on the shtools library for SH-related functions. It also provides utilities to convert between different SH coordinate conventions (e.g., bip2017, sfsNet) and the one adopted here. Training data is organized into separate train, validation, and test file lists, while the rendering-based data generation pipeline resides in a separate repository.

### Notes

- We provide one example input image and seven example lighting configurations in the `data` directory.

- Different methods may adopt different coordinate systems for Spherical Harmonics (SH).  
  If you use SH lighting from external sources, you may need to convert the coordinate system accordingly.

- The coordinate system used in this project follows the convention of **shtools**.

- We provide a utility script `utils_normal.py` (located in the `utils` directory) to help convert SH coefficients from:
  - [bip2017](https://gravis.dmi.unibas.ch/PMM/data/bip/)
  - [sfsNet](https://senguptaumd.github.io/SfSNet/)  
  into the coordinate system used in this project.

- To use `utils_normal.py`, you must install **shtools**.

- This code is provided for **research purposes only**.

---
# Project Structure
```text
.
├── candidiate_config
│   ├── candidate1
│   │   ├── rotate_light_04_0.txt
│   │   ├── rotate_light_04.txt
│   │   └── rotate_light_06.txt
│   ├── candidate2
│   │   ├── rotate_light_03_0.txt
│   │   ├── rotate_light_03_1.txt
│   │   ├── rotate_light_03.txt
│   │   ├── rotate_light_04_0.txt
│   │   └── rotate_light_04.txt
│   └── candidate3
│       ├── rotate_light_02.txt
│       ├── rotate_light_03.txt
│       ├── rotate_light_04_0.txt
│       └── rotate_light_04.txt
├── data
│   ├── example_light
│   │   ├── rotate_light_00.txt
│   │   ├── rotate_light_01.txt
│   │   ├── rotate_light_02.txt
│   │   ├── rotate_light_03.txt
│   │   ├── rotate_light_04.txt
│   │   ├── rotate_light_05.txt
│   │   └── rotate_light_06.txt
│   ├── example_light_original
│   │   ├── rotate_light_00.txt
│   │   ├── rotate_light_01.txt
│   │   ├── rotate_light_02.txt
│   │   ├── rotate_light_03.txt
│   │   ├── rotate_light_04.txt
│   │   ├── rotate_light_05.txt
│   │   └── rotate_light_06.txt
│   ├── obama.jpg
│   ├── test.lst
│   ├── train.lst
│   └── val.lst
├── model
│   ├── defineHourglass_1024_gray_skip_matchFeature.py
│   └── defineHourglass_512_gray_skip.py
├── obama.jpg
├── README.md
├── requirment.txt
├── result
│   ├── light_00.png
│   ├── light_01.png
│   ├── light_02.png
│   ├── light_03.png
│   ├── light_04.png
│   ├── light_05.png
│   ├── light_06.png
│   ├── obama_00.jpg
│   ├── obama_01.jpg
│   ├── obama_02.jpg
│   ├── obama_03.jpg
│   ├── obama_04.jpg
│   ├── obama_05.jpg
│   └── obama_06.jpg
├── test
│   └── test_light_condition.py
├── trained_model
│   ├── trained_model_03.t7
│   └── trained_model_1024_03.t7
└── utils
    ├── testNetwork_demo_1024.py
    ├── testNetwork_demo_512.py
    ├── utils_normal.py
    ├── utils_SH.py
    └── utils_shtools.py
```

# Requirements

To run this project, please make sure the following prerequisites are met:

- **Python 3.10**
- All required packages listed in `requirement.txt`

We strongly recommend using a virtual environment to avoid dependency conflicts.

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python3.10 -m venv dpr-env
source dpr-env/bin/activate      # On Linux / macOS
# dpr-env\Scripts\activate       # On Windows
```
2. install python packages
```bash
pip install --upgrade pip
pip install -r requirement.txt 
```
--- 

# Brightness Configuration File

- The directory `data/example_light_original` contains multiple lighting configuration files.  
  Each configuration defines a specific lighting condition, including variations in illumination angle and light intensity.

- These configuration files are used to simulate how the portrait appears under different lighting directions and strengths.

- Detailed explanations of the configuration file parameters are provided in the following section.

This configuration file contains settings that include 9 coefficients related to lighting.

```text
1.084125496282453138e+00
-4.642676300617166185e-01
2.837846795150648915e-02
6.765292733937575687e-01
-3.594067725393816914e-01
4.790996460111427574e-02
-2.280054643781863066e-01
-8.125983081159608712e-02
2.881082012687687932e-01
```
- The **first coefficient** represents the ambient light. This value determines the overall base brightness of the image regardless of direction.
- The **next 3 coefficients** represent directional linear lighting along the x, y, and z axes respectively. These values specify the main light direction (left/right, up/down, front/back) shining onto the face.
  - **Light along X axis**: Controls the directional    light intensity along the X axis (left ↔ right).
  - **Light along Y axis**: Controls the directional light intensity along the Y axis (up ↔ down).
  - **Light along Z axis**: Controls the directional light intensity along the Z axis (front ↔ back).

- The **last 5 coefficients** represent complex lighting details used for more accurate simulation of light diffusion and softer shadows on curved surfaces (such as a human face).
  - **Coefficient dependent on x·y**: Represents lighting variations influenced by the interaction between the X and Y axes.
  - **Coefficient dependent on y·z**: Represents lighting variations influenced by the interaction between the Y and Z axes.
  - **Coefficient dependent on 3z² − 1**: Controls spherical lighting deformation related to the Z‑axis curvature component.
  - **Coefficient dependent on x·z**: Represents lighting variations influenced by the interaction between the X and Z axes.
  - **Coefficient dependent on x² − y²**: Describes lighting differences caused by the relative contribution of the X and Y axes.

These coefficients work together to create a realistic lighting effect in the rendered image.

---
# Running the Project

- In the project repository, a set of **unit tests** has been provided that can be used to generate example outputs.

- These tests are located in the corresponding test directory of the project.

- The main methods for producing output are:
  - `test_network_512_demo`
  - `test_network_1024_demo`

- These two methods demonstrate how to run the model and obtain relit portrait results.

- The other test methods in the directory provide additional functionality and examples.  
  You can understand their usage by reading the code.

- For generating initial outputs and testing the pipeline, running the two demo methods above is sufficient.

---

### Data Preparation
We publish the code for data preparation, please find it in (https://github.com/zhhoper/RI_render_DPR).

### Citation
If you use this code for your research, please consider citing:
```
@InProceedings{DPR,
  title={Deep Single Portrait Image Relighting},
  author = {Hao Zhou and Sunil Hadap and Kalyan Sunkavalli and David W. Jacobs},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
