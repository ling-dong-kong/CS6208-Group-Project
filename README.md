# CS6208: Group Project
> **Student:** Kong Lingdong (A0260240X)<br>
> **Title:** "Robust Point Cloud Classification with Dual Dynamic Graph Consistency"<br>
> **Time:** AY 2022-2023, Semester II

## About
This repository contains the code and implementation details of the Group Project assignment for the CS6208 course. In this assignment, we propose a dual dynamic graph consistency framework, dubbed `Dual-DGC`, for robust point cloud classification under out-of-distribution corruption scenarios.

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

### Training
To train `Dual-DGC` on ModelNet40, run the following command:
```

```
### Evaluation
To evaluate `Dual-DGC` on ModelNet40, first set the folder path for the trained model, as `$OUTPUT_DIR`, then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 eval.py --eval --ckpt $OUTPUT_DIR/model.t7 --model dual_dgc --if_attn
```
### Robustness Test
To evaluate `Dual-DGC` on all corruption sets in ModelNet-C, run the following command:
```

```


## Main Result

| Method | mCE | Scale | Jitter | Drop-G | Drop-L | Add-G | Add-L | Rotate | OA |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| DGCNN        | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.926 |
| PointNet     | 1.422 | 1.266 | 0.642 | 0.500 | 1.072 | 2.980 | 1.593 | 1.902 | 0.907 |
| PointNet++   | 1.072 | 0.872 | 1.177 | 0.641 | 1.802 | 0.614 | 0.993 | 1.405 | 0.930 |
| RSCNN        | 1.130 | 1.074 | 1.171 | 0.806 | 1.517 | 0.712 | 1.153 | 1.479 | 0.923 |
| SimpleView   | 1.047 | 0.872 | 0.715 | 1.242 | 1.357 | 0.983 | 0.844 | 1.316 | 0.939 |
| PAConv       | 1.104 | 0.904 | 1.465 | 1.000 | 1.005 | 1.085 | 1.298 | 0.967 | 0.936 |
| PCT          | 0.925 | 0.872 | 0.870 | 0.528 | 1.000 | 0.780 | 1.385 | 1.042 | 0.930 |
| **Dual-DGC** |

## Reference

- [(Wang, et al., 2019)](https://arxiv.org/abs/1801.07829) Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. "Dynamic graph cnn for learning on point clouds." *ACM Transactions On Graphics*, 38(5): 1–12, 2019.
- [(We, et al., 2015)](https://arxiv.org/abs/1406.5670) Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., and Xiao, J. "3d shapenets: A deep representation for volumetric shapes." In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1912–1920, 2015.


