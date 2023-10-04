# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .ray import generate_rays


nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552, 
    1583610, 364372, 2346381, 582961, 4829021, 
    14073691, 191019309, 6249651, 55095657, 
    58484771, 193834360, 131378779
])
dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]


def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
        dtype=np.float32,
        count=-1).reshape(-1, 3)
    
    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:,2]
    return coords, depth_label

def load_seg_label(img_file_path, gt_path, img_size=[900,1600], mode='lidarseg'):
    if mode=='lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1],coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map

def get_sensor_transforms(cam_info, cam_name):
    w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
    # sweep sensor to sweep ego
    sensor2ego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sensor2ego_tran = torch.Tensor(
        cam_info['cams'][cam_name]['sensor2ego_translation'])
    sensor2ego = sensor2ego_rot.new_zeros((4, 4))
    sensor2ego[3, 3] = 1
    sensor2ego[:3, :3] = sensor2ego_rot
    sensor2ego[:3, -1] = sensor2ego_tran
    # sweep ego to global
    w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
    ego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    ego2global_tran = torch.Tensor(
        cam_info['cams'][cam_name]['ego2global_translation'])
    ego2global = ego2global_rot.new_zeros((4, 4))
    ego2global[3, 3] = 1
    ego2global[:3, :3] = ego2global_rot
    ego2global[:3, -1] = ego2global_tran

    return sensor2ego, ego2global


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, 
                use_rays=False,
                semantic_gt_path=None,
                depth_gt_path=None,
                aux_frames=[-1,1],
                max_ray_nums=0,
                wrs_use_batch=False,
                **kwargs):
        super().__init__(**kwargs)
        self.use_rays = use_rays
        self.semantic_gt_path = semantic_gt_path
        self.depth_gt_path = depth_gt_path
        self.aux_frames = aux_frames
        self.max_ray_nums = max_ray_nums

        if wrs_use_batch:   # compute with batch data
            self.WRS_balance_weight = None
        else:               # compute with total dataset
            self.WRS_balance_weight = torch.exp(0.005 * (nusc_class_nums.max() / nusc_class_nums - 1))

        self.dynamic_class = torch.tensor(dynamic_class)


    def get_rays(self, index):
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                label_seg = seg_map[coor[:,1], coor[:,0]]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(label_seg))
                time_ids[time_id].append(idx)
                idx += 1
        
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)

        # generate rays for all frames
        rays = generate_rays(
            coors, label_depths, label_segs, sensor2keyegos, intrins, 1600, 900, 
            max_ray_nums=self.max_ray_nums, 
            time_ids=time_ids, 
            dynamic_class=self.dynamic_class, 
            balance_weight=self.WRS_balance_weight)
        return rays

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        # generate rays for rendering supervision
        if self.use_rays:
            rays = self.get_rays(index)
            input_dict['rays'] = rays
        else:
            input_dict['rays'] = torch.zeros((1))
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        return self.occ_eval_metrics.count_miou()
