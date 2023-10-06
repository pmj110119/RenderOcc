# RenderOcc

### [paper](https://arxiv.org/abs/2309.09502) | [video](https://www.youtube.com/watch?v=UcdXM3FNLAc)

![demo](assets/demo.gif)

## Note: Codebase under active development

We are in the process of organizing and documenting the complete codebase. 

## Get Started
### Installation and Data Preparation
1. Create a conda environment and install pytorch:
    ```
    conda create -n monodetr python=3.8
    conda activate monodetr

    conda install pytorch torchvision cudatoolkit
    # We adopt torch 1.10.1+cu111
    ```

2. Clone and install this repo:
    ```
    git clone https://github.com/pmj110119/RenderOcc.git
    cd Renderocc

    pip install -r requirements.txt
    pip install -v -e .
    ```

3. Prepare dataset
    step 1: Download nuScenes dataset from [HERA](https://www.nuscenes.org/download) on `./data/nuscenes`.
    step 2: Download (only) the 'gts' from [HERA](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) on `./data/nuscenes` and arrange the folder as:
    ```
    └── nuscenes
    ├── v1.0-trainval (step 1)
    ├── sweeps  (step 1)
    ├── samples (step 1)
    └── gts     (step 2)
    ```
    step 3: Create the pkl:
    ```
    python tools/create_data_occ.py
    ```
    step 4: Create 2D GT for RenderOcc:
    ```
    python tools/gen_data/gen_depth_gt.py
    python tools/gen_data/gen_seg_gt_from_lidarseg.py
    ```

### Train
```
# multiple gpu
./tools/dist_train.sh configs/renderocc/renderocc-7frame.py 8
```
( Please download pretrain model for BEVStereo from [HERE](https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw?pwd=npe1) and put it on `assets/` )

### test
```
# multiple gpu
./tools/dist_test.sh configs/renderocc/renderocc-7frame.py $checkpoint 8
```
    