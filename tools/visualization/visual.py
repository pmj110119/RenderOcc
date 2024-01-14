import os
import cv2
import glob
import pickle 
import argparse
import numpy as np
from vis_tool import vis_one_frame, merge_images, merge_all


def mask_sky(occ, n=3):
    occ[:,:,-n:]=17   
    return occ

def mask_ego_car(occ):
    occ[93:107,95:105,4:8]=17 
    return occ

def visual_ego_car(occ):
    occ[96:103,98:102,4:7]=4
    return occ

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple argparse example.')
    parser.add_argument('dump_dir', type=str)
    args = parser.parse_args()

    save_dir = args.dump_dir+'_visual'
    os.makedirs(save_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.dump_dir,'*.npy'))
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    for file in files:
        frame_idx = int(file.split('/')[-1].split('.')[0])

        with open(file, 'rb') as f:
            occ = np.load(f)[0]
        occ = mask_sky(occ, n=3)     # omit areas over 3m height for better visualization
        occ = mask_ego_car(occ) # omit ego-car

        occ_captures = []
        for cam in ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right', 'top']:
            if 'top' in cam:
                occ = mask_sky(occ, n=6)    # omit areas
                occ = visual_ego_car(occ)   # visualize ego car with cube
            occ_capture = vis_one_frame(occ, 'tools/visualization/viewpoint_params/cam_%s.json'%cam.lower(), manual=False)
            if 'back' in cam:
                occ_capture = cv2.flip(occ_capture, 1)
            occ_captures.append(occ_capture)
            

        occ_camrea_view = merge_images(occ_captures[:6])
        occ_top_view = occ_captures[6]
        camera_visual = cv2.imread(file.replace('npy', 'png'))

        occ_visual = merge_all(camera_visual, occ_camrea_view, occ_top_view)
        cv2.imwrite('%s/%d.png'%(save_dir, frame_idx), occ_visual)
            
