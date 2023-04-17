# CS6208: Group Project
> **Student:** Kong Lingdong (A0260240X)<br>
> **Title:** "Robust Point Cloud Classification with Dual Dynamic Graph CNNs"<br>
> **Time:** AY 2022-2023, Semester II

## About
This repository contains the code and implementation details of the Group Project assignment for the CS6208 course. In this assignment, we propose a dual dynamic graph CNNs, dubbed `Dual-DGC`, for robust point cloud classification under out-of-distribution corruption scenarios.

## Installation
This codebase is tested with `torch==1.10.0` with `CUDA 11.3`. In order to successfully reproduce the results reported, we recommend to follow the exact same configuation. However, similar versions that came out lately should be good as well.

- Step 1: Create Enviroment
```
conda create -n dual_dgc python=3.7
```
- Step 2: Activate Enviroment
```
conda activate dual_dgc
```
- Step 3: Install PyTorch
```
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch
```
- Step 4: Install Necessary Libraries
```
pip install numpy sklearn glob h5py plyfile torch_scatter
```

## Data Preparation

### ModelNet40
To prepare the `ModelNet40` dataset, download the data from [[this](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)] official webpage and put the folder into `./data/`.
```
└── data 
      └── modelnet40_ply_hdf5_2048
          |── ply_data_train*.h5 (#5)
          |── ply_data_test*.h5 (#2)
          |── shape_names.txt
          |── train_files.txt
          |── test_files.txt
          |── ply_data_train*.json (#5)
          └── ply_data_test*.json (#2)
```

### ModelNet-C
To prepare the `ModelNet-C` dataset, download the data from [[this](https://drive.google.com/file/d/1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm/view?usp=sharing)] Google Drive link and put the folder into `./data/`.
```
└── data 
      └── modelnet_c
          |── clean.h5
          |── add_global*.h5 (#5)
          |── add_local*.h5 (#5)
          |── dropout_global*.h5 (#5)
          |── dropout_local*.h5 (#5)
          |── jitter*.h5 (#5)
          |── rotate*.h5 (#5)
          └── scale*.h5 (#5)
  ```

## Getting Started


## Main Result


## Reference

- [(Wang, et al., 2019)](https://arxiv.org/abs/1801.07829) Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. "Dynamic graph cnn for learning on point clouds." *ACM Transactions On Graphics*, 38(5): 1–12, 2019.
- [(We, et al., 2015)](https://arxiv.org/abs/1406.5670) Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., and Xiao, J. "3d shapenets: A deep representation for volumetric shapes." In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1912–1920, 2015.


