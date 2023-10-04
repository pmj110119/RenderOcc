import os
import cv2
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_coo
from torch_efficient_distloss import flatten_eff_distloss

from .utils import Raw2Alpha, Alphas2Weights, ub360_utils_cuda, silog_loss


# OpenOccupancy
# nusc_class_frequencies = np.array([2242961742295, 25985376, 1561108, 28862014, 196106643, 15920504,
#                 2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
#                 1446141335, 1724391378])

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

def sample_ray(ori_rays_o, ori_rays_d, step_size, scene_center, scene_radius, bg_len, world_len, bda, **render_kwargs):
    rays_o = (ori_rays_o - scene_center) / scene_radius       # normalization
    rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
    N_inner = int(2 / (2+2*bg_len) * world_len / step_size) + 1
    N_outer = N_inner//15   # hardcode: 15
    b_inner = torch.linspace(0, 2, N_inner+1)
    b_outer = 2 / torch.linspace(1, 1/64, N_outer+1)
    t = torch.cat([
        (b_inner[1:] + b_inner[:-1]) * 0.5,
        (b_outer[1:] + b_outer[:-1]) * 0.5,
    ]).to(rays_o)
    ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]

    norm = ray_pts.norm(dim=-1, keepdim=True)
    inner_mask = (norm<=1)
    ray_pts = torch.where(
        inner_mask,
        ray_pts,
        ray_pts / norm * ((1+bg_len) - bg_len/norm)
    )

    # reverse bda-aug 
    ray_pts = bda.matmul(ray_pts.unsqueeze(-1)).squeeze(-1)
    return ray_pts, inner_mask.squeeze(-1), t



from mmdet.models import HEADS
@HEADS.register_module()
class NerfHead(nn.Module):
    def __init__(self, 
            point_cloud_range,
            voxel_size,
            scene_center=None,
            radius=39,
            step_size=0.5, 
            use_depth_sup=False,
            balance_cls_weight=True,
            weight_depth=1.0,
            weight_semantic=1.0,
            weight_entropy_last=0.01,   
            weight_distortion=0.01,
            alpha_init=1e-6,
            fast_color_thres=1e-7,
            ):
        super().__init__()
        self.weight_entropy_last = weight_entropy_last
        self.weight_distortion = weight_distortion

        xyz_min = torch.Tensor(point_cloud_range[:3])
        xyz_max = torch.Tensor(point_cloud_range[3:])
        xyz_range = (xyz_max - xyz_min).float()
        self.bg_len = (xyz_range[0]//2-radius)/radius
        # import ipdb;ipdb.set_trace()
        self.radius = radius


        self.register_buffer('scene_center', ((xyz_min + xyz_max) * 0.5))
        self.register_buffer('scene_radius', torch.Tensor([radius, radius, radius]))

        self.step_size = step_size
        self.use_depth_sup = use_depth_sup
        
        z_ = xyz_range[2]/xyz_range[0]
        self.register_buffer('xyz_min', torch.Tensor([-1-self.bg_len, -1-self.bg_len, -z_]))
        self.register_buffer('xyz_max', torch.Tensor([1+self.bg_len, 1+self.bg_len, z_]))
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('--> Set density bias shift to', self.act_shift)

        
        self.voxel_size = voxel_size/radius
        self.voxel_size_ratio = torch.tensor(1.0)
        self.world_size = torch.Tensor([200,200,16]).long()
        self.world_len = self.world_size[0].item()

        self.fast_color_thres = fast_color_thres
        self.weight_depth = weight_depth
        self.weight_semantic = weight_semantic
        self.depth_loss = silog_loss()

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001))
        else:
            self.class_weights = torch.ones(17)/17 

    def render_one_scene(self,
            rays_o_tr,
            rays_d_tr,
            bda,
            density,
            semantic,
            mask=None,
        ):
        if mask is not None:
            rays_o = rays_o_tr[mask]
            rays_d = rays_d_tr[mask]
        else:
            rays_o = rays_o_tr.reshape(-1, 3)
            rays_d = rays_d_tr.reshape(-1, 3)
        device = rays_o.device

        # sample points on rays
        ray_pts, inner_mask, t = sample_ray(
            ori_rays_o=rays_o, ori_rays_d=rays_d, 
            step_size=self.step_size,
            scene_center=self.scene_center, 
            scene_radius=self.scene_radius, 
            bg_len=self.bg_len, 
            world_len=self.world_len, 
            bda=bda,
        )
        
        torch.cuda.empty_cache()
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        dist_thres = (2+2*self.bg_len) / self.world_len * self.step_size * 0.95
        dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        mask[:, 1:] |= ub360_utils_cuda.cumdist_thres(dist, dist_thres)
        ray_pts = ray_pts[mask]
        inner_mask = inner_mask[mask]

        N_ray = len(rays_o)
        t = t[None].repeat(N_ray,1)[mask]
        ray_id = ray_id[mask.flatten()].to(device)
        step_id = step_id[mask.flatten()].to(device)

        # rays sampling
        shape = ray_pts.shape[:-1]
        xyz = ray_pts.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        density = F.grid_sample(density.unsqueeze(0).unsqueeze(1), ind_norm, mode='bilinear', align_corners=True)
        density = density.reshape(1, -1).T.reshape(*shape) 

        
        semantic = semantic.permute(3,0,1,2).unsqueeze(0)
        num_classes = semantic.shape[1]
        semantic = F.grid_sample(semantic, ind_norm, mode='bilinear', align_corners=True)
        semantic = semantic.reshape(num_classes, -1).T.reshape(*shape, num_classes)

        alpha = self.activate_density(density, interval=0.5) 
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            semantic = semantic[mask]

        # compute accumulated transmittance
        N_ray = len(rays_o)
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id.to(alpha.device), N_ray)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            inner_mask = inner_mask[mask]
            t = t[mask]
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            semantic = semantic[mask]


        s = 1 - 1/(1+t)  # [0, inf] => [0, 1]
        results = {
            'alphainv_last': alphainv_last,
            'weights': weights,
            'ray_id': ray_id,
            's': s,
            't': t,
            'N_ray': N_ray,
            'num_classes': num_classes,
            'density': density,
            'semantic': semantic,
        }
        return results
    

    def compute_loss(self, results):
        losses = {}
        if self.use_depth_sup:
            depth_loss = self.depth_loss(results['render_depth']+1e-7, results['target_depth'])
            losses['loss_render_depth'] = depth_loss * self.weight_depth
        
        target_semantic = results['target_semantic']
        semantic = results['render_semantic']
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.type_as(semantic), reduction="mean"
        )
        semantic_loss = criterion(semantic, target_semantic.long())
        losses['loss_render_semantic'] = semantic_loss * self.weight_semantic
    

        if self.weight_entropy_last > 0:
            pout = results['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            losses['loss_sdf_entropy'] = self.weight_entropy_last * entropy_last_loss

        if self.weight_distortion > 0:
            n_max = len(results['t'])
            loss_distortion = flatten_eff_distloss(results['weights'], results['s'], 1/n_max, results['ray_id'])
            losses['loss_sdf_distortion'] =  self.weight_distortion * loss_distortion
        return losses

    def render_depth(self, results):
        depth = segment_coo(
                        src=(results['weights'] * results['s']),
                        index=results['ray_id'],
                        out=torch.zeros([results['N_ray']]).to(results['weights'].device),
                        reduce='sum') + 1e-7
        return depth * self.radius  

    def render_semantic(self, results):
        semantic = segment_coo(
                        src=(results['weights'].unsqueeze(-1) * results['semantic']),
                        index=results['ray_id'],
                        out=torch.zeros([results['N_ray'], results['num_classes']]).to(results['weights'].device),
                        reduce='sum')
        return semantic
    
    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)


    def forward(self, density, semantic, rays=None, bda=None, **kwargs):
        gt_depths = rays[..., 2]
        gt_semantics = rays[..., 3]
        ray_o = rays[..., 4:7]
        ray_d = rays[..., 7:10]

        losses = {}
        for batch_id in range(rays.shape[0]): 
            torch.cuda.empty_cache()
            rays_o_tr = ray_o[batch_id]
            rays_d_tr = ray_d[batch_id]
            
            ## ================  depth & semantic supervision  ===================
            gt_depth = gt_depths[batch_id]
            gt_semantic = gt_semantics[batch_id]
            gt_depth[gt_depth>52] = 0   
            mask = gt_depth>0 
            target_depth = gt_depth[mask]
            target_semantic = gt_semantic[mask]
            
            results = {}
            results['target_semantic'] = target_semantic
            results['target_depth'] = target_depth

            results.update(
                self.render_one_scene(
                    rays_o_tr=rays_o_tr,
                    rays_d_tr=rays_d_tr,
                    bda=bda[batch_id],
                    mask = mask,
                    density=density[batch_id],
                    semantic=semantic[batch_id],
                )
            )

            # render depth & semantic
            if self.use_depth_sup:
                results['render_depth'] = self.render_depth(results)
            results['render_semantic'] = self.render_semantic(results)
            # compute loss
            loss_single = self.compute_loss(results)
            for key in loss_single:
                if key in losses:
                    losses[key] = losses[key] + loss_single[key]
                else:
                    losses[key] = loss_single[key]
        for key in losses:
            losses[key] = losses[key] / density.shape[0]
        return losses

