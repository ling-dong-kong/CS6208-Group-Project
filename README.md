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


## Getting Started


## Main Result


## Reference

- [(Wang, et al., 2019)](https://arxiv.org/abs/1801.07829) Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. "Dynamic graph cnn for learning on point clouds." *ACM Transactions On Graphics*, 38(5): 1–12, 2019.
- [(We, et al., 2015)](https://arxiv.org/abs/1406.5670) Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., and Xiao, J. "3d shapenets: A deep representation for volumetric shapes." In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1912–1920, 2015.


