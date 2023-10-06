import enum
import os
from multiprocessing import Pool

import mmcv
import numpy as np
import cv2
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import copy
np.random.seed(0)
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar2ego_translation,
    lidar2ego_rotation,
    ego2global_translation,
    ego2global_rotation,
    sensor2ego_translation, 
    sensor2ego_rotation,
    cam_ego2global_translation,
    cam_ego2global_rotation,
    cam_intrinsic,
    min_dist: float = 0.0,
    ):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar2ego_rotation).rotation_matrix)
    pc.translate(np.array(lidar2ego_translation))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(ego2global_rotation).rotation_matrix)
    pc.translate(np.array(ego2global_translation))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego2global_translation))
    pc.rotate(Quaternion(cam_ego2global_rotation).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(sensor2ego_translation))
    pc.rotate(Quaternion(sensor2ego_rotation).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    labels = pc.points[3, :]
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         cam_intrinsic,
                         normalize=True)
    
    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    # mask = np.logical_and(mask, labels != 17)
    points = points[:, mask]
    coloring = coloring[mask]
    labels = labels[mask].astype(np.int32)
    return points, coloring, labels


info_path_train = './data/nuscenes/occ_infos_temporal_train.pkl'
info_path_val = './data/nuscenes/occ_infos_temporal_val.pkl'

# data3d_nusc = NuscMVDetData()

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]



colors = np.random.randint(0, 255, size=(40, 3))
def get_voxel_coords(arr):
    x, y, z = arr.shape
    coords = np.indices((x, y, z)).transpose(1, 2, 3, 0)
    return coords

def draw_points(img, pts_img, label):
    for i in range(pts_img.shape[1]):
        x, y = pts_img[:, i]
        color = colors[label[i]]
        cv2.circle(img, (x, y), 3, color.tolist(), -1)
    return img

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

from nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

label_name = {
    0: 'noise', 
    1: 'animal', 
    2: 'human.pedestrian.adult', 
    3: 'human.pedestrian.child', 
    4: 'human.pedestrian.construction_worker', 
    5: 'human.pedestrian.personal_mobility', 
    6: 'human.pedestrian.police_officer', 
    7: 'human.pedestrian.stroller', 
    8: 'human.pedestrian.wheelchair', 
    9: 'movable_object.barrier', 
    10: 'movable_object.debris', 
    11: 'movable_object.pushable_pullable', 
    12: 'movable_object.trafficcone', 
    13: 'static_object.bicycle_rack', 
    14: 'vehicle.bicycle', 
    15: 'vehicle.bus.bendy', 
    16: 'vehicle.bus.rigid', 
    17: 'vehicle.car', 
    18: 'vehicle.construction', 
    19: 'vehicle.emergency.ambulance', 
    20: 'vehicle.emergency.police', 
    21: 'vehicle.motorcycle', 
    22: 'vehicle.trailer', 
    23: 'vehicle.truck', 
    24: 'flat.driveable_surface', 
    25: 'flat.other', 
    26: 'flat.sidewalk', 
    27: 'flat.terrain', 
    28: 'static.manmade', 
    29: 'static.other', 
    30: 'static.vegetation', 
    31: 'vehicle.ego'}

label_map = {
    'animal':0, 
    'human.pedestrian.personal_mobility':0, 
    'human.pedestrian.stroller':0, 
    'human.pedestrian.wheelchair':0, 
    'movable_object.debris':0,
    'movable_object.pushable_pullable':0, 
    'static_object.bicycle_rack':0, 
    'vehicle.emergency.ambulance':0, 
    'vehicle.emergency.police':0, 
    'noise':0, 
    'static.other':0, 
    'vehicle.ego':0,
    'movable_object.barrier':1, 
    'vehicle.bicycle':2,
    'vehicle.bus.bendy':3,
    'vehicle.bus.rigid':3,
    'vehicle.car':4,
    'vehicle.construction':5,
    'vehicle.motorcycle':6,
    'human.pedestrian.adult':7,
    'human.pedestrian.child':7,
    'human.pedestrian.construction_worker':7,
    'human.pedestrian.police_officer':7,
    'movable_object.trafficcone':8,
    'vehicle.trailer':9,
    'vehicle.truck':10,
    'flat.driveable_surface':11,
    'flat.other':12,
    'flat.sidewalk':13,
    'flat.terrain':14,
    'static.manmade':15,
    'static.vegetation':16
}

filter_mask = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

label_merged_map = {}
for idx in label_name:
    name = label_name[idx]
    idx_merged = label_map[name]
    label_merged_map[idx] = idx_merged
print(label_merged_map)

label_merged_map = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26:
13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0}

names = set()

save_folder = os.path.join('./data/nuscenes/', 'seg_gt_merge17_val') 
mmcv.mkdir_or_exist(save_folder)

def worker(info):
    visual=False
    save_folder = os.path.join('./data/nuscenes/', 'seg_gt_merge17_val') 
    lidar_path = info['lidar_path']
    
    # for i, cam_key in enumerate(cam_keys):
    #     # if cam_key!=cam_keys[5]: continue
    #     file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]

    #     if 'n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__' in file_name and '15372908' in file_name:
    #         import ipdb;ipdb.set_trace()
    #         print(2)
    #     else:
    #         return
    #     # if file_name!='n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537290893428113.jpg':
    #     #     print(file_name)
    #     #     return
    #     # else:
            



    # import ipdb;ipdb.set_trace()
    # nusc.render_pointcloud_in_image(info['token'],
    #                                 pointsensor_channel='LIDAR_TOP',
    #                                 camera_channel='CAM_FRONT',
    #                                 render_intensity=False,
    #                                 show_lidarseg=True,
    #                                 # filter_lidarseg_labels=[22, 23, 24],
    #                                 show_lidarseg_legend=True)
    
    points = np.fromfile(lidar_path,
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]

    lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                            nusc.get('lidarseg', 
                                                nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                                                )['filename'])
    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    if 'merge' in save_folder:
        points_label_merge = np.zeros_like(points_label)
        for key in label_merged_map:
            points_label_merge[points_label==key] = label_merged_map[key]
    points[:,3] = points_label_merge

    # if (points==7).sum()>0:
    #     print((points_label==key).sum())
    #     import ipdb;ipdb.set_trace()
    #     print(111)

    # mask = points_label!=0      # 消除others类的影响（lidarseg中将所有五花八门的稀有类合并为others、不参与评测，cvpr这个比赛参与评测有点搞。。）
    # points = points[mask]

    lidar2ego_translation = info['lidar2ego_translation']
    lidar2ego_rotation = info['lidar2ego_rotation']
    ego2global_translation = info['ego2global_translation']
    ego2global_rotation = info['ego2global_rotation']
    for i, cam_key in enumerate(cam_keys):
        # if cam_key!=cam_keys[5]: continue
        file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        # if file_name != 'n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800853912460.jpg':
        #     continue
        # name = file_name.split('-')[0]
        # names.add(name)
        # print(name)
        # continue
        sensor2ego_translation = info['cams'][cam_key]['sensor2ego_translation']
        sensor2ego_rotation = info['cams'][cam_key]['sensor2ego_rotation']
        cam_ego2global_translation = info['cams'][cam_key]['ego2global_translation']
        cam_ego2global_rotation = info['cams'][cam_key]['ego2global_rotation']
        cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
        img = mmcv.imread(
            os.path.join(info['cams'][cam_key]['data_path']))
        pts_img, depth, label = map_pointcloud_to_image(
            points.copy(), img, 
            copy.deepcopy(lidar2ego_translation), 
            copy.deepcopy(lidar2ego_rotation), 
            copy.deepcopy(ego2global_translation),
            copy.deepcopy(ego2global_rotation),
            copy.deepcopy(sensor2ego_translation), 
            copy.deepcopy(sensor2ego_rotation), 
            copy.deepcopy(cam_ego2global_translation), 
            copy.deepcopy(cam_ego2global_rotation),
            copy.deepcopy(cam_intrinsic))
        
        
        
        # exit()
      

        np.concatenate([pts_img[:2, :].T, label[:,None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(save_folder, f'{file_name}.bin'))
        # import ipdb;ipdb.set_trace()
        if visual:
            mmcv.mkdir_or_exist(os.path.join('./data', 'seg_gt_visual'))
            png_name = os.path.join('./data', 'seg_gt_visual', file_name)
            img_drawed = draw_points(img, pts_img[:2,:].astype(np.int), label)
            cv2.imwrite(png_name, img_drawed)
        
    # import ipdb;ipdb.set_trace()

from tqdm import tqdm

from multiprocessing import Pool


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    # infos = mmcv.load(info_path_train)['infos']
    # lens = len(infos)

    # idx = args.idx
    # start, end = (lens//4)*idx, (lens//4)*(idx+1)
    # print(start, end, lens)
    # # if idx==4:
    # #     with Pool(8) as p:
    # #         p.map(worker, infos[start:])
    # # else:
    # #     with Pool(8) as p:
    # #         p.map(worker, infos[start:end])

    # for info in tqdm(infos,total=len(infos)):
    #     worker(info)
    #     for cam_key in cam_keys:
    #         file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
    #         # print(file_name)





    infos = mmcv.load(info_path_val)['infos']
    print(len(infos))
    with Pool(6) as p:
        p.map(worker, infos)


        
    
