
# I3D_Light

A light version of the two-stream I3D model which uses a lower resolution video (spatial resolution = 112x112 compared to 224x224) of a longer temporal range (temporal range = 128 frames compared to 32 frames) RGB branch to capture the motion features. <br>
<br>
This code is based on Miracleyoo's [Trainable-i3d-pytorch](https://github.com/miracleyoo/Trainable-i3d-pytorch)
<br><br>
<img src="I3D-Light.png">
&nbsp;           &nbsp;Figure 1: Architecture of proposed two-stream I3D Light.

<br><br>

## Setup

```shell
git clone https://github.com/Ruikang07/I3D_Light.git
cd i3d_light

conda create --name i3d_light python=3.7
conda activate i3d_light
pip install -r requirements.txt
```

## Dataset Folder Structure

```
dataset_in_imgs
├── classes.txt
├── train
│   ├── action1
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   └── ...
│   ├── action2
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── img ...
│   │   └── ...
│   └── ...
│
├── val
│   ├── action1
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   └── ...
│   ├── action2
│   │   ├── video1
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── ...
│   │   ├── video2
│   │   │       ├── img1.png
│   │   │       ├── img2.png
│   │   │       └── img ...
│   │   └── ...
│   └── ...
│
└── json_file_dir
