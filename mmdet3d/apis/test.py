# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import time
import shutil
import pickle
import tempfile
import numpy as np
import os.path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

def single_gpu_test(model, data_loader, dump=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        # import ipdb;ipdb.set_trace()
        with torch.no_grad():
            result = model.module(return_loss=False, rescale=True, **data)
        results.extend(result)

        if dump:
            scene_name = data['img_metas'][0].data[0][0]['scene_name']
            frame_idx = data['img_metas'][0].data[0][0]['frame_idx']
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def reverse_norm(image_list):
    for i in range(len(image_list)):
        img = image_list[i]
        img = (img-img.min())/(img.max()-img.min()) * 255
        image_list[i] = img.astype(np.uint8)
    return image_list

def merge_images(image_list, gap_width=10, size=[500,800]):
    assert len(image_list)==6, '<merge_images> Must be 6 cameras!'
    H, W = size
    # create an empty canvas with 2 rows and 3 columns
    merged_image_height = 2 * H + gap_width*1
    merged_image_width = 3 * W + gap_width*2
    merged_image = np.zeros((merged_image_height, merged_image_width, 3), dtype=np.uint8)

    # Merge images into canvas, and insert a 4-pixel black edge between two images
    for i in range(2):
        for j in range(3):
            y1 = i * (H + gap_width)
            y2 = (i + 1) * (H + gap_width) - gap_width
            x1 = j * (W + gap_width)
            x2 = (j + 1) * (W + gap_width) - gap_width

            img = cv2.resize(image_list[i * 3 + j], (W,H))
            merged_image[y1:y2, x1:x2] = img
    return merged_image


def multi_gpu_test(model, data_loader, dump_dir=None, tmpdir=None, gpu_collect=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if dump_dir is not None:
            scene_name = data['img_metas'][0].data[0][0]['scene_name']
            frame_idx = data['img_metas'][0].data[0][0]['frame_idx']

            # dump occupancy prediction
            save_path = os.path.join(dump_dir, scene_name)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, '%.3d.npy'%frame_idx), result)

            # dump images
            imgs = data['img_inputs'][0][0][0]
            TN, C, H, W = imgs.shape
            imgs = imgs.reshape(6, TN//6, C, H, W)[:,0,...]    # select key frame only
            imgs = imgs.cpu().numpy().transpose(0,2,3,1)
            image_list = [imgs[0], imgs[1], imgs[2],
                            imgs[5][:,::-1,:], imgs[4][:,::-1,:], imgs[3][:,::-1,:]
                            ]
            image_list = reverse_norm(image_list)
            img_merged = merge_images(image_list, gap_width=10, size=[500,800])
            cv2.imwrite(os.path.join(save_path, '%.3d.png'%frame_idx), img_merged)

        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
                
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
