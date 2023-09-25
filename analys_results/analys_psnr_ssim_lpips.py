import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torch

def get_metric(from_img_file, gt_img_file):

    print(from_img_file)
    from_img_ori = cv2.imread(from_img_file)
    from_img_ori = cv2.cvtColor(from_img_ori, cv2.COLOR_BGR2RGB)
    from_img = (np.array(from_img_ori) ).astype(np.float32)

    gt_img_ori = cv2.imread(gt_img_file)
    gt_img_ori = cv2.cvtColor(gt_img_ori, cv2.COLOR_BGR2RGB)
    gt_img = (np.array(gt_img_ori) ).astype(np.float32)

    psnr = compare_psnr(
                    from_img, gt_img,data_range=255)
    ssim = compare_ssim(
                    from_img, gt_img,channel_axis=2,data_range=255)
    
    from_img_torch = (from_img_ori / 128.) - 1.0
    from_img_torch = torch.tensor(from_img_torch.transpose(2,0,1)).unsqueeze(0).float()
    gt_img_torch = (gt_img_ori / 128.) - 1.0
    gt_img_torch = torch.tensor(gt_img_torch.transpose(2,0,1)).unsqueeze(0).float()
    lpips = loss_fn_vgg(from_img_torch, gt_img_torch)
    lpips = lpips.detach().numpy()[0,0,0,0]

    return psnr, ssim, lpips

if __name__ == '__main__':
    
    train_view_list = ['21320027','21320030','21320035']

    gt_path = './data/catch_sequence/final_render_img'
    ours_path = './fit_res/analys_res/view_3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    save_param = {}
    file_name_list = []
    ours_psnr_list,ours_ssim_list,ours_lpips_list  = [], [], []
    anerf_psnr_list,anerf_ssim_list,anerf_lpips_list  = [], [], []
    ibrnet_psnr_list,ibrnet_ssim_list,ibrnet_lpips_list  = [], [], []
    infonerf_psnr_list,infonerf_ssim_list,infonerf_lpips_list  = [], [], []

    for obj_name in os.listdir(gt_path):
        obj_path = os.path.join(gt_path, obj_name)
        for frame_name in os.listdir(obj_path):
            frame_path = os.path.join(obj_path, frame_name)
            mask_path = os.path.join(frame_path, 'MASK')
            for file_name in os.listdir(mask_path):
                view_name = file_name.split('.')[0].split('_')[1]
                if view_name in train_view_list:
                    continue
                png_name = file_name.replace('jpeg', 'png')
                gt_img = os.path.join(mask_path, file_name)
                our_img = os.path.join(ours_path, '12', obj_name, frame_name, 'render_12', file_name)

                ours_psnr, ours_ssim, ours_lpips = get_metric(our_img, gt_img)
                cur_file_name = obj_name + '+' + frame_name + '+' + file_name
                file_name_list.append(cur_file_name)
                ours_psnr_list.append(ours_psnr)
                ours_ssim_list.append(ours_ssim)
                ours_lpips_list.append(ours_lpips)

    ours_psnr_list = np.array(ours_psnr_list)
    ours_ssim_list = np.array(ours_ssim_list)
    ours_lpips_list = np.array(ours_lpips_list)
    print(ours_psnr_list.shape[0])

    ours_psnr_list_mean = ours_psnr_list.mean()
    ours_ssim_list_mean = ours_ssim_list.mean()
    ours_lpips_list_mean = ours_lpips_list.mean()
   
    print('     psnr,     ssim,     lpips')
    print('ours: ', ours_psnr_list_mean,ours_ssim_list_mean,ours_lpips_list_mean)

        