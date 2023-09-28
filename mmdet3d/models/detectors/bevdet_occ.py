# Copyright (c) Phigent Robotics. All rights reserved.
from bdb import set_trace
from .bevdet import BEVStereo4D
import torch.nn.functional as F
import torch
import cv2
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .. import builder
import matplotlib.pyplot as plt
import matplotlib
colors_map = np.array(
    [
        [0,   0,   0],  # 0 undefined
        [255, 158, 0],  # 1 barrier         障碍物  浅橘色
        [0, 0, 230],    # 2 bicycle  Blue
        [200, 0, 0],   # 3 bus             公交车  深绿色
        [220, 20, 60],  # 4 car             小汽车  深红色
        [200, 200, 200],   # 5 construction_vehicle  工程车  
        [255, 140, 0],  # 6 motorcycle      摩托车  深橘色
        [233, 150, 70], # 7 pedestrian      行人    
        [255, 61, 99],  # 8 traffic_cone    锥桶    Red
        [112, 128, 144],# 9 trailer         拖车    Slategrey
        [222, 184, 135],# 10 truck          卡车    茶棕色
        [100, 100, 100],    # 11 driveable_surface 
        [165, 42, 42],  # 12 other_flat     其他    深棕色
        [50, 50, 50],  # 13 sidewalk   
        [75, 0, 75], # 14 terrain
        [255, 0, 0], # 15 manmade
        [0, 175, 0], # 16 vegetation
        [255,255,255], # 17 free
    ])


# CVPR workshop
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])



@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 balance_cls_weight=False,
                 use_depth_gt=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)

        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.use_depth_gt = use_depth_gt

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(18)/18

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego   # 取的是第一帧、第一个相机时间戳下的pose作为key
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()     # B, T, N, 4, 4     #这里得到了所有时序、相机帧 keyego的pose！

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,     # camera2keyego 
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),     # ida_aug，不同相机aug不同，同一相机不同时间戳共享aug参数
            post_trans.view(B, self.num_frame, N, 3)        # ida_aug，不同相机aug不同，同一相机不同时间戳共享aug参数
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame, depth_gt=None):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        

        bev_feat, depth = self.img_view_transformer(
             [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input], 
             metas,
             depth_gt)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(self,
                         img_inputs,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, curr2adjsensor = img_inputs
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None

        if self.use_depth_gt:
            B, N, C, H, W = imgs[0].shape
            T = len(imgs)
            gt_depths = kwargs['gt_depths']
            if isinstance(gt_depths, list):
                gt_depths = gt_depths[0]
            gt_depths = gt_depths.reshape(B,N,T,H,W).permute(0,2,1,3,4) # B,T,N,H,W

        for fid in range(self.num_frame-1, -1, -1):  
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0   
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames   
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(     
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)

                # import ipdb;ipdb.set_trace()
                depth_gt = None
                if self.use_depth_gt:
                    depth_gt = gt_depths[:, fid, ...]
                    if False:
                        depth_gt2 = gt_depths[:, fid, ...][0,2]
                        coords = torch.nonzero(depth_gt2)
                        depth_gt2 = depth_gt2[coords[:, 0], coords[:, 1]]
                        coords = coords.cpu().numpy()

                        src = img[0,2].permute(1,2,0)
                        src = 255*(src-src.min())/(src.max()-src.min())
                        src = src.cpu().numpy().astype(np.uint8)
                        src = np.ascontiguousarray(src)

                        cmap = matplotlib.cm.get_cmap("viridis")
                        coloured_intensity = 255*cmap(depth_gt2.cpu().numpy()/30)

                        for i in range(coords.shape[0]):
                            _ = cv2.circle(src, (int(coords[i,1])+1,int(coords[i,0])+1), 3, coloured_intensity[i])

                        # for i in range(coords.shape[0]):
                        #     src[coords[:, 0], coords[:, 1]] = coloured_intensity[i][:3]
                        # src[mask] = depth_gt2[mask].item()
                        cv2.imwrite('zz.png', src)
           
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame, depth_gt)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv

        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:  
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame

    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = preds.reshape(-1, self.num_classes)

        # compute loss
        loss_ = dict()
        loss_occ = self.loss_occ(preds, voxel_semantics,)
        loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_inputs = self.prepare_inputs(img, stereo=True)
        img_feats, _ = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        res = [occ_res]
        return res

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):

        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses



