import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler



def generate_frustum(img, cam2ego, cam_intrinsic):
    device = imgs.device
    rgb_tr = imgs.permute(0,2,3,1)
    poses = sensor2keyegos
    Ks = cam_intrinsic
    N, H, W = rgb_tr.shape[:3]

    rays_o_tr = torch.zeros([N, H, W, 3], device=device)  
    rays_d_tr = torch.zeros([N, H, W, 3], device=device)   
    viewdirs_tr = torch.zeros([N, H, W, 3], device=device) 
    for cam_id in range(N):
        c2w = poses[cam_id]
        K = Ks[cam_id]
        rays_o_tmp, rays_d_tmp, viewdirs_tmp = get_rays_of_a_view(
                H=H, W=W, K=K, 
                c2w=c2w, ndc=False, 
                inverse_y=True, flip_x=False, flip_y=False
                )
        rays_o_tr[cam_id].copy_(rays_o_tmp.to(device))
        rays_d_tr[cam_id].copy_(rays_d_tmp.to(device))
        viewdirs_tr[cam_id].copy_(viewdirs_tmp.to(device))
        del rays_o_tmp, rays_d_tmp, viewdirs_tmp
    return rays_o_tr, rays_d_tr, viewdirs_tr, rgb_tr




def get_rays(i, j, K, c2w, inverse_y=True):
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)   
    return rays_o, rays_d, viewdirs



def pts2ray(X, Y, coor, label_depth, label_seg, c2w, cam_intrinsic):
    rays_o, rays_d, viewdirs = get_rays(X, Y, K=cam_intrinsic,c2w=c2w)
    coor = coor.long()
    x,y = coor[:,1], coor[:,0]
    rays_o, rays_d, viewdirs = rays_o[y,x], rays_d[y,x], viewdirs[y,x]
    return torch.cat([
        coor, label_depth.unsqueeze(1), label_seg.unsqueeze(1),  # 0-1, 2, 3
        rays_o, rays_d, viewdirs        # 4:7,7:10,10:13
        ], dim=1
    )
    

def generate_rays(depth_map, seg_map, c2w, intrins, img_size=(900,1600), max_ray_nums=0, time_ids=None, dynamic_class=None, balance_weight=None, weight_adj=0.3, weight_dyn=0.0, use_wrs=True):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Args:
        depth_map (Nx1): depth GT of pixels
        seg_map (Nx1): semantic GT of pixels
        c2w : camera to ego
        intrins : camera intrins matrix
        width : width of RGB image
        height : height of RGB image
        max_ray_nums : 
        time_ids (Tx1) : 
        balance_weight : 
        weight_adj : 
        weight_dyn : 
        use_wrs : 
    """

    device = c2w.device
    height, width = img_size
    X, Y = torch.meshgrid(
        torch.linspace(0, width-1, width, device=device),
        torch.linspace(0, height-1, height, device=device))  # pytorch's meshgrid has indexing='ij'
    X = X.t().float()+0.5   # W
    Y = Y.t().float()+0.5   # H
    
    
    batch_rays = []
    for batch_id in range(depth_map.shape[0]):
        # generate rays
        rays = []
        ids = []
        for time_id in time_ids:    # multi frames
            for idx in time_ids[time_id]: # multi cameras of single frame
                i = idx[batch_id]
                coors = torch.nonzero(depth_map[batch_id][i] != -1)
                label_depths = depth_map[batch_id][i][coors[:,0], coors[:,1]]
                label_segs = seg_map[batch_id][i][coors[:,0], coors[:,1]]
                ray = pts2ray(X, Y, coors, label_depths, label_segs, c2w[batch_id][i], intrins[batch_id][i])
                rays.append(ray)
                ids.append(time_id)

        # Weighted Rays Sampling
        if not use_wrs:
            rays = torch.cat(rays, dim=0)
        else:
            weights = []
            if balance_weight is None:  # use batch data to compute balance_weight ( rather than the total dataset )
                classes = torch.cat([ray[:,3] for ray in rays])
                class_nums = torch.Tensor([0]*17)
                for class_id in range(17): 
                    class_nums[class_id] += (classes==class_id).sum().item()
                balance_weight = torch.exp(0.005 * (class_nums.max() / class_nums - 1))

            for i in range(len(rays)):
                # wrs-a
                ans = 1.0 if ids[i]==0 else weight_adj
                weight_t = torch.full((rays[i].shape[0],), ans, device=device)
                if ids[i]!=0:
                    mask_dynamic = (dynamic_class[batch_id] == rays[i][:, 3, None]).any(dim=-1)
                    weight_t[mask_dynamic] = weight_dyn
                # wrs-b
                weight_b = balance_weight[batch_id][rays[i][..., 3].long()]

                weight = weight_b * weight_t
                weights.append(weight)

            rays = torch.cat(rays, dim=0)
            weights = torch.cat(weights, dim=0)
            if max_ray_nums[batch_id]!=0 and rays.shape[0]>max_ray_nums[batch_id]:
                sampler = WeightedRandomSampler(weights, num_samples=max_ray_nums[batch_id].item(), replacement=False)
                rays = rays[list(sampler)]
            batch_rays.append(rays)
    return torch.stack(batch_rays)
        
