
## NuScenes
- Prepare Data
step 1: Download nuScenes V1.0 full dataset data from [HERA](https://www.nuscenes.org/download) on `./data/nuscenes`.

step 2: Download (only) the 'gts' from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D)


step 3: Create the pkl files:
```
python tools/create_data_bevdet.py
```

step 4: Create 2D GT for RenderOcc:

```
python tools/gen_data/gen_depth_gt.py
python tools/gen_data/gen_seg_gt_from_lidarseg.py
```

- Download Pretrained model weights

```
mkdir ckpts
cd ckpts & wget https://github.com/pmj110119/storage/releases/download/v1/bevdet-stbase-4d-stereo-512x1408-cbgs.pth
```

**Folder structure**
```
RenderOcc
├── mmdet3d/
├── tools/
├── configs/
├── ckpts/
│   ├── TODO.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s occupancy gts to this location
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── bevdetv2-nuscenes_infos_val.pkl
|   |   ├── bevdetv2-nuscenes_infos_train.pkl
```