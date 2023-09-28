import os
from typing import List, Optional, Tuple
import numpy as np
import requests
import torch
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, Sampler
import random
import math
import cv2
import open3d as o3d

def load_ply(pth):
    mesh = o3d.io.read_triangle_mesh(pth)
    vts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    return vts, faces

def get_rays_xy(image,mask, mask_xy, n_rays_per_image, threshold =0.4):
    H, W = image.shape[:2]
    mask_x, mask_y = mask_xy
    mask_x = torch.LongTensor(mask_x)
    mask_y = torch.LongTensor(mask_y)
    sele_mask_num = int(n_rays_per_image * threshold)
    sele_mask_num = min(sele_mask_num, mask_x.shape[0])
    sele_id = np.random.randint(mask_x.shape[0], size=sele_mask_num)
    sele_mask_x = mask_x[sele_id]
    sele_mask_y = mask_y[sele_id]
    sele_img_part1 = image[sele_mask_x, sele_mask_y, :]
    sele_mask_part1 = mask[sele_mask_x, sele_mask_y, :]
    rays_part1 = torch.cat((sele_mask_y[...,None], sele_mask_x[...,None]), -1)
    sele_other_num = n_rays_per_image - sele_mask_num
    sele_other = np.random.rand(sele_other_num, 2)
    sele_other_x = torch.tensor(sele_other[:,0] * H).long().cpu()
    sele_other_y = torch.tensor(sele_other[:,1] * W).long().cpu()
    sele_img_part2 = image[sele_other_x, sele_other_y,:]
    sele_mask_part2 = mask[sele_other_x, sele_other_y,:]
    rays_part2 = torch.cat((sele_other_y[...,None], sele_other_x[...,None]), -1)
    rays_xy = torch.cat((rays_part1.cpu(), rays_part2.cpu()), 0)
    rays_xy = rays_xy.float()
    rays_xy[:,0] = (rays_xy[:,0] - (W/2.)) / (H/2.)
    rays_xy[:,1] = (rays_xy[:,1] - (H/2.)) / (H/2.)
    rays_xy *= -1
    true_rgb = torch.cat((sele_img_part1, sele_img_part2),0)
    true_mask = torch.cat((sele_mask_part1, sele_mask_part2),0)
    return rays_xy, true_rgb, true_mask

def get_rays_xy_mask(image, mask, mask_xy, n_rays_per_image):
    H, W = mask.shape[:2]
    mask_x_list, mask_y_list = mask_xy
    x_min = mask_x_list.min()
    x_max = mask_x_list.max()
    y_min = mask_y_list.min()
    y_max = mask_y_list.max()
    c_len = np.sqrt(n_rays_per_image)
    x_st = min(x_min+c_len, x_max)
    x_ed = max(x_min+c_len, x_max)
    y_st = min(y_min+c_len, y_max)
    y_ed = max(y_min+c_len, y_max)
    end_x = random.randint(x_st, x_ed)
    end_y = random.randint(y_st, y_ed)
    x_id = np.arange(end_x-c_len, end_x)
    y_id = np.arange(end_y-c_len, end_y)
    mask_x, mask_y = np.meshgrid(x_id,y_id,indexing='xy')
    mask_x = mask_x.reshape(-1)
    mask_y = mask_y.reshape(-1)
    true_rgb = image[mask_x, mask_y,:]
    true_mask = mask[mask_x,mask_y,:]
    mask_x = (mask_x - (H/2.)) / (H/2.)
    mask_y = (mask_y - (W/2.)) / (H/2.)
    rays_xy = np.concatenate((mask_y[...,None], mask_x[...,None]), -1)
    rays_xy *= -1
    rays_xy = torch.FloatTensor(rays_xy)
    return rays_xy, true_rgb, true_mask

def get_bone_length(ori_T_pose_21):
    length_list = []
    father_list = [0,0,0, 0, 0,1,5,9,13,17,2,6,10,14,18,3,7,11,15,19]
    child_list =  [1,5,9,13,17,2,6,10,14,18,3,7,11,15,19,4,8,12,16,20]
    for i in range(20):
        fater_id = father_list[i]
        child_id = child_list[i]
        ori_length = np.sqrt(((ori_T_pose_21[child_id] - ori_T_pose_21[fater_id])**2).sum())
        length_list.append(ori_length)
    return np.array(length_list)

def get_pose_from_param(label,image):
    R = label['cam_R']
    T = label['cam_T'] 
    fx_ndc = label['fx_ndc'] 
    fy_ndc = label['fy_ndc'] 
    px_ndc = label['px_ndc'] 
    py_ndc = label['py_ndc']
    h = label['H']
    w = label['W']
    s = min(h,w)
    h = h-1
    w = w-1
    s = s-1
    K = np.eye(4)
    fx = -1.0 * fx_ndc * (s ) / 2.0
    fy = -1.0 * fy_ndc * (s ) / 2.0
    cx = -1.0 * px_ndc * (s ) / 2.0 + (w ) / 2.0
    cy = -1.0 * py_ndc * (s ) / 2.0 + (h ) / 2.0
    K[0, 0], K[1, 1] = fx, fy
    K[0,2], K[1,2] = cx, cy
    pose = np.eye(4)
    pose[:3,:3] = R.transpose()
    pose[:3,3] = T
    return K, pose

class TrainDataLoad(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        n_rays_per_image,
        data_type,
        model_type
    ):
        all_imgs = []
        all_masks = []
        all_verts = []
        all_T_pose_21 = []
        all_mask_xy = []
        all_R = []
        all_T = []
        all_Ro = []
        all_To = []
        all_focal_length = []
        all_principal_point = []
        all_bone_length = []
        all_2djoint = []
        self.n_rays_per_image = n_rays_per_image
        
        if model_type == 'obj':
            IMG_path = os.path.join(data_root, 'MASK')
            PARAM_path = os.path.join(data_root,'PARAM_266')
            pred_pose_path = os.path.join(data_root,'pred_objpose_8view')
            param_name_list = os.listdir(PARAM_path)
            param_name_list.sort()
            if 'bean' in data_root:
                file_path = os.path.join(data_root,'bean_ours.ply')
            elif 'meat' in data_root:
                file_path = os.path.join(data_root,'meat_ours.ply')
            elif 'box' in data_root:
                file_path = os.path.join(data_root,'box_ours.ply')
            elif 'cup' in data_root:
                file_path = os.path.join(data_root,'cup_ours.ply')
            vert_model, _ = load_ply(file_path)
            vert_model /= 1000.0
            vert_model = np.array(vert_model[::50,:])
            for param_file_name in param_name_list:
                img_file_name = param_file_name.replace('pickle','jpeg')
                img_file = os.path.join(IMG_path,img_file_name)
                file_name = param_file_name.split('.')[0]
                cid, view_name = file_name.split('_')
                param_file = os.path.join(PARAM_path, param_file_name)
                pose_file = os.path.join(pred_pose_path, cid + '.txt')
                with open(param_file,'rb') as f:
                    param = pickle.load(f)
                cosypose = np.loadtxt(pose_file).astype(np.float32)
                Ro = cosypose[:3,:3]
                To = cosypose[:3,3]
                cur_img_ori = param['color_img']
                mask = (cur_img_ori > 0).all(axis=-1)[..., None].astype(np.uint8)
                mask_xy = np.where((mask[:,:,0] > 0))
                mask = mask * 255
                all_imgs.append(cur_img_ori)
                all_masks.append(mask)
                all_mask_xy.append(mask_xy)
                all_R.append(param['cam_R'])
                all_T.append(param['cam_T'])
                all_Ro.append(Ro)
                all_To.append(To)
                focal_length = np.array((param['fx_ndc'],param['fy_ndc'])) 
                all_focal_length.append(focal_length) 
                principal_point = np.array((param['px_ndc'],param['py_ndc']))
                all_principal_point.append(principal_point)
                all_verts.append(vert_model)
                all_T_pose_21.append(np.zeros((21,3)))
                all_2djoint.append(np.zeros((21,2)))
                all_bone_length.append(np.zeros((21,2)))

        elif model_type == 'hand':
            IMG_path = os.path.join(data_root,'IMG')
            PARAM_path = os.path.join(data_root,'PARAM_266')
            mppose_path = os.path.join(data_root, 'mppose_3d')
            img_name = os.listdir(IMG_path)
            img_name.sort()
            ori_param_file = os.path.join(data_root,'t_pose_mppose.pickle')
            with open(ori_param_file,'rb') as f:
                ori_param = pickle.load(f)
            ori_T_pose_21 = ori_param['T_pose_21']
            cur_bone_length = get_bone_length(ori_T_pose_21)
            for img_file_name in img_name:
                file_name = img_file_name.split('.')[0]
                cid, view_name = file_name.split('_')
                param_file_name = img_file_name.replace('.jpeg', '.pickle')
                img_file = os.path.join(IMG_path, img_file_name)
                param_file = os.path.join(PARAM_path, param_file_name)
                pose_file = os.path.join(mppose_path, cid + '.pickle')
                with open(param_file,'rb') as f:
                    param = pickle.load(f)
                cur_img_ori = param['color_img']
                mask = (cur_img_ori > 0).all(axis=-1)[..., None].astype(np.uint8)
                mask_xy = np.where((mask[:,:,0] > 0))
                cur_img_ori = cur_img_ori * mask
                mask = mask * 255
                all_imgs.append(cur_img_ori)
                all_masks.append(mask)
                all_mask_xy.append(mask_xy)
                all_R.append(param['cam_R'])
                all_T.append(param['cam_T'])
                all_Ro.append(np.eye(3))
                all_To.append(np.zeros(3))
                with open(pose_file,'rb') as f:
                    joint_3d = pickle.load(f) 
                all_verts.append(joint_3d)
                focal_length = np.array((param['fx_ndc'],param['fy_ndc'])) 
                all_focal_length.append(focal_length) 
                principal_point = np.array((param['px_ndc'],param['py_ndc']))
                all_principal_point.append(principal_point)
                all_T_pose_21.append(ori_T_pose_21)
                all_bone_length.append(cur_bone_length)
                intrinsics, pose = get_pose_from_param(param,cur_img_ori)
                pro_mat = intrinsics[:3,:3] @ pose[:3,:4]
                pro_2d_joint = (pro_mat[:3,:3] @ joint_3d.T).T + pro_mat[:,3]
                pro_2d_joint[:,:2] /= pro_2d_joint[:,2:]
                all_2djoint.append(pro_2d_joint[:,:2])

        self.img_num = len(all_R)
        print(self.img_num)
        images = torch.FloatTensor((np.array(all_imgs) / 255.).astype(np.float32))
        masks = torch.FloatTensor((np.array(all_masks) / 255.).astype(np.float32))
        self.images = images
        self.masks = masks
        self.all_R = torch.FloatTensor(np.array(all_R))
        self.all_T = torch.FloatTensor(np.array(all_T))
        self.all_Ro = torch.FloatTensor(np.array(all_Ro))
        self.all_To = torch.FloatTensor(np.array(all_To))
        self.all_focal_length = torch.FloatTensor(np.array(all_focal_length))
        self.all_principal_point = torch.FloatTensor(np.array(all_principal_point))
        self.all_verts = torch.FloatTensor(np.array(all_verts))
        self.all_T_pose_21= torch.FloatTensor(np.array(all_T_pose_21))
        self.all_bone_length= torch.FloatTensor(np.array(all_bone_length))
        self.all_mask_xy = all_mask_xy
        self.all_2djoint = torch.LongTensor(all_2djoint)
        print("load finish")
        
    def __len__(self):
        return self.img_num 

    def getitem(self, index):
        image = self.images[index]
        mask = self.masks[index]
        R = self.all_R[index]
        T = self.all_T[index]
        Ro = self.all_Ro[index]
        To = self.all_To[index]
        focal_length = self.all_focal_length[index]
        principal_point = self.all_principal_point[index]
        verts = self.all_verts[index]
        bone_length = self.all_bone_length[index]
        patch_sample = get_rays_xy_mask(image, mask,self.all_mask_xy[index],self.n_rays_per_image)
        random_sample = get_rays_xy(image, mask,self.all_mask_xy[index],self.n_rays_per_image)
        T_pose_21 = self.all_T_pose_21[index]
        return image,mask,R,T,focal_length,principal_point,Ro,To,verts,\
            random_sample, patch_sample,T_pose_21,bone_length,index

    def __getitem__(self, index):
        return self.getitem(index)

class TestDataLoad(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        data_type,
        model_type,
    ):

        all_imgs = []
        all_masks = []
        all_verts = []
        all_trans = []
        all_T_pose_21 = []
        all_t0 = []
        all_focal_length = []
        all_principal_point = []
        all_R = []
        all_T = []
        all_Ro = []
        all_To = []
        param_file_list = []
        if model_type == 'obj':
            PARAM_path = os.path.join(data_root,'PARAM_266')
            pred_pose_path = os.path.join(data_root,'pred_objpose_8view')
            param_name_list = os.listdir(PARAM_path)
            param_name_list.sort()
            ori_bone_length = np.zeros((20,3))
            for param_file_name in param_name_list:
                if '21320034' not in param_file_name:
                    continue
                file_name = param_file_name.split('.')[0]
                cid, view_name = file_name.split('_')
                param_file = os.path.join(PARAM_path, param_file_name)
                with open(param_file,'rb') as f:
                    param = pickle.load(f)
                param_file_list.append(param_file_name)
                all_R.append(param['cam_R'])
                all_T.append(param['cam_T'])
                all_Ro.append(param['obj_R'])
                all_To.append(param['obj_T'])
                focal_length = np.array((param['fx_ndc'],param['fy_ndc'])) 
                all_focal_length.append(focal_length) 
                principal_point = np.array((param['px_ndc'],param['py_ndc']))
                all_principal_point.append(principal_point)
                all_verts.append(np.zeros((21,3)))
                all_T_pose_21.append(np.zeros((21,3)))

        elif model_type == 'hand':
            IMG_path = os.path.join(data_root,'IMG')
            PARAM_path = os.path.join(data_root,'PARAM_266')
            mppose_path = os.path.join(data_root, 'mppose_3d')
            img_name = os.listdir(IMG_path)
            img_name.sort()
            ori_param_file = os.path.join(data_root,'t_pose_mppose.pickle')
            with open(ori_param_file,'rb') as f:
                ori_param = pickle.load(f)
            ori_T_pose_21 = ori_param['T_pose_21']
            ori_bone_length = get_bone_length(ori_T_pose_21)
            for img_file_name in img_name:
                if data_type == 'syn':
                    param_file_name = img_file_name.replace('.png', '.pickle')
                else:
                    param_file_name = img_file_name.replace('.jpeg', '.pickle')
                param_file = os.path.join(PARAM_path, param_file_name)
                file_name = img_file_name.split('.')[0]
                cid, view_name = file_name.split('_')                
                with open(param_file,'rb') as f:
                    param = pickle.load(f)
                all_R.append(param['cam_R'])
                all_T.append(param['cam_T'])
                all_Ro.append(np.eye(3))
                all_To.append(np.zeros(3))
                joint_3d = param['joint3d_21'] 
                param_file_list.append(param_file_name)
                all_verts.append(joint_3d)
                focal_length = np.array((param['fx_ndc'],param['fy_ndc']))
                all_focal_length.append(focal_length)
                principal_point = np.array((param['px_ndc'],param['py_ndc']))
                all_principal_point.append(principal_point)
                all_T_pose_21.append(ori_T_pose_21)

        self.img_num = len(all_R)
        print(self.img_num)
        self.all_R = torch.FloatTensor(np.array(all_R))
        self.all_T = torch.FloatTensor(np.array(all_T))
        self.all_Ro = torch.FloatTensor(np.array(all_Ro))
        self.all_To = torch.FloatTensor(np.array(all_To))
        self.all_focal_length = torch.FloatTensor(np.array(all_focal_length))
        self.all_principal_point = torch.FloatTensor(np.array(all_principal_point))
        self.all_verts = torch.FloatTensor(np.array(all_verts))
        self.all_T_pose_21 = torch.FloatTensor(np.array(all_T_pose_21))
        self.param_file_list = param_file_list
        self.ori_bone_length= torch.FloatTensor(ori_bone_length)

    def __len__(self):
        return self.img_num

    def getitem(self, index):
        return self.all_R[index],self.all_T[index],self.all_focal_length[index],self.all_principal_point[index],\
               self.all_Ro[index],self.all_To[index],\
               self.all_T_pose_21[index],self.ori_bone_length,\
               index,\
               self.param_file_list[index], self.all_verts[index]

    def __getitem__(self, index):
        return self.getitem(index)

class RayImageSampler(Sampler):
    '''
    TODO: does this work with ConcatDataset?
    TODO: handle train/val
    '''

    def __init__(self, data_source, N_images=1024,
                 N_iter=None, generator=None, device='cpu'):
        self.data_source = data_source
        self.N_images = N_images
        self._N_iter = N_iter

    def __iter__(self):
        sampler_iter = iter(range(len(self.data_source)))
        batch = []
        for i in range(self._N_iter):
            for n_img in range(self.N_images):
                batch.append(i + n_img)
            # return and clear batch cache
            yield np.sort(batch)
            batch = []

    def __len__(self):
        return self._N_iter

class fit_single_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        view_num,
        fit_type,
    ):  

        if view_num == '8':
            view_name_list = ['21320018','21320027','21320028','21320029',\
                              '21320030','21320034','21320035','21320036',]
        elif view_num == '6':
            view_name_list = ['21320018','21320027','21320028',\
                              '21320034','21320035','21320036',]
        elif view_num == '3':
            view_name_list = ['21320027','21320030','21320035',]

        test_view_name_list = ['21320018']

        self.img_groups = []
        self.mask_groups = []
        self.param_groups = []
        self.test_param_groups = []
        self.name_groups = []
        self.objpose_groups = []
        self.handpose_groups = []
        self.proj_groups = []
        self.fitparam_groups = []
        self.T_pose_21_groups = []
        self.bone_length_groups = []
        self.palm_angle_groups = []
        self.obj_verts_groups = []
        self.obj_faces_groups = []
        self.hand_model_path = []
        self.obj_model_path = []
        self.save_base_path = []
        self.test_name_groups = []
        obj_list = os.listdir(data_root)
        obj_list.sort()
        for obj_name in obj_list:
            per, obj = obj_name.split("_")
            obj_path = os.path.join(data_root, obj_name)
            frame_list = os.listdir(obj_path)
            frame_list.sort()
            for frame_name in frame_list:
                frame_path = os.path.join(obj_path, frame_name)
                img_path = os.path.join(frame_path, 'MASK')
                mesh_file = os.path.join(frame_path, obj + '_ours.ply')
                obj_verts, obj_faces = load_ply(mesh_file)
                obj_verts /= 1000.0
                for frame_id in range(2000):
                    cur_img_list = []
                    cur_mask_list = []
                    cur_param_list = []
                    cur_img_name_list = []
                    cur_objpose_list = []
                    cur_handjoint_list = [] 
                    cur_proj_list = []
                    cur_fitparam_list = []
                    cur_test_param_list = []
                    cur_test_img_name_list = []
                    test_file = os.path.join(img_path, str(frame_id) + '_21320018.jpeg')
                    if not os.path.exists(test_file):
                        continue
                    hand_model_path = './exp/'+ per + '/wmask_realhand'
                    obj_model_path = './exp/'+ obj + '/wmask_realobj'
                    for view_name in view_name_list:
                        file_name = str(frame_id) + '_' + view_name
                        img_file = os.path.join(img_path, file_name + '.jpeg')
                        param_file = os.path.join(frame_path, 'PARAM_266', file_name + '.pickle')
                        ori_param_file = os.path.join(frame_path,per + '_tmppose.pickle')
                        with open(ori_param_file,'rb') as f:
                            ori_param = pickle.load(f)
                        T_pose_21 = ori_param['T_pose_21']
                        bone_length = get_bone_length(T_pose_21)
                        cur_img_ori = cv2.imread(img_file)
                        cur_img_ori = cv2.resize(cur_img_ori, (266,230))
                        mask = (cur_img_ori > 10).all(axis=-1)[..., None].astype(np.uint8) * 255
                        img = (np.array(cur_img_ori) / 255.).astype(np.float32)
                        mask = (np.array(mask) / 255.).astype(np.float32)
                        with open(param_file,'rb') as f:
                            param = pickle.load(f)
                        if fit_type == '1':
                            pred_3djoint_file = os.path.join(frame_path, 'pred_joint3d_'+str(len(view_name_list))+'view', str(frame_id) + '.pickle')
                            pred_objpose_file = os.path.join(frame_path, 'pred_objpose_'+str(len(view_name_list))+'view', str(frame_id) + '.txt')
                            with open(pred_3djoint_file,'rb') as f:
                                joint3d_param = pickle.load(f)
                            pred_joint3d = joint3d_param['pred_joint_3d'].astype(np.float32)
                            obj_pose = np.loadtxt(pred_objpose_file).astype(np.float32)
                            save_base_path = os.path.join('./fit_res', 'view_' + str(len(view_name_list)), '1', obj_name, frame_name)

                        elif fit_type == '12':
                            pred_pose_path = os.path.join('./fit_res', 'view_'  + str(len(view_name_list)), '1', obj_name, frame_name, 'pose_1')
                            pred_pose_file = os.path.join(pred_pose_path, str(frame_id) + '.pickle')
                            if not os.path.exists(pred_pose_file):
                                print(pred_pose_file + ' dose not exist!')
                            with open(pred_pose_file,'rb') as f:
                                pred_param = pickle.load(f)
                            pred_joint3d = pred_param['pred_joint3d'].astype(np.float32)
                            fit_1_Ro = pred_param['pred_Ro'].astype(np.float32)
                            fit_1_To = pred_param['pred_To'].astype(np.float32)
                            obj_pose = np.eye(4)
                            obj_pose[:3,:3] = fit_1_Ro
                            obj_pose[:3,3] = fit_1_To
                            save_base_path = os.path.join('./fit_res', 'view_'  + str(len(view_name_list)), '12', obj_name, frame_name)
                        
                        height = cur_img_ori.shape[0]
                        width = cur_img_ori.shape[1]
                        s = min(height, width)
                        R = param['cam_R'].T
                        T = param['cam_T'] 
                        fx_ndc = param['fx_ndc'] 
                        fy_ndc = param['fy_ndc'] 
                        px_ndc = param['px_ndc'] 
                        py_ndc = param['py_ndc']
                        K = np.eye(3)
                        fx = -1.0 * fx_ndc * (s-1 ) / 2.0 
                        fy = -1.0 * fy_ndc * (s-1 ) / 2.0
                        cx = -1.0 * px_ndc * (s-1 ) / 2.0 + (width-1 ) / 2.0
                        cy = -1.0 * py_ndc * (s-1 ) / 2.0 + (height-1 ) / 2.0
                        K[0, 0], K[1, 1] = fx, fy
                        K[0,2], K[1,2] = cx, cy
                        view_trans = np.zeros((3, 4))
                        view_trans[:3,:3] = R
                        view_trans[:3,3] = T
                        proj = K @ view_trans
                        cur_img_list.append(img)
                        cur_mask_list.append(mask)
                        cur_img_name_list.append(file_name + '.jpeg')
                        cur_handjoint_list.append(pred_joint3d)
                        cur_objpose_list.append(obj_pose)
                        cur_param_list.append(param)
                        cur_proj_list.append(proj)
                    for view_name in test_view_name_list:
                        file_name = str(frame_id) + '_' + view_name
                        param_file = os.path.join(frame_path, 'PARAM_266', file_name + '.pickle')
                        with open(param_file,'rb') as f:
                            param = pickle.load(f)
                        cur_test_param_list.append(param)
                        cur_test_img_name_list.append(file_name + '.jpeg')
                        
                    if len(cur_param_list) != 0:
                        self.img_groups.append(cur_img_list)
                        self.mask_groups.append(cur_mask_list)
                        self.param_groups.append(cur_param_list)
                        self.name_groups.append(cur_img_name_list)
                        self.handpose_groups.append(cur_handjoint_list)
                        self.objpose_groups.append(cur_objpose_list)
                        self.proj_groups.append(cur_proj_list)
                        self.T_pose_21_groups.append(T_pose_21)
                        self.bone_length_groups.append(bone_length)
                        self.obj_verts_groups.append(obj_verts)
                        self.obj_faces_groups.append(obj_faces)
                        self.hand_model_path.append(hand_model_path)
                        self.obj_model_path.append(obj_model_path)
                        self.save_base_path.append(save_base_path)
                        self.test_param_groups.append(cur_test_param_list)
                        self.test_name_groups.append(cur_test_img_name_list)

        self.img_groups = np.array(self.img_groups)
        self.mask_groups = np.array(self.mask_groups)
        self.handpose_groups = np.array(self.handpose_groups)
        self.objpose_groups = np.array(self.objpose_groups)
        self.proj_groups = np.array(self.proj_groups)
        self.T_pose_21_groups = np.array(self.T_pose_21_groups)
        self.bone_length_groups = np.array(self.bone_length_groups)
        self.obj_verts_groups = self.obj_verts_groups
        print(self.img_groups.shape[0])

    def __len__(self):
        return len(self.img_groups)

    def getitem(self, index):
        image_group = torch.FloatTensor(self.img_groups[index])
        mask_group = torch.FloatTensor(self.mask_groups[index])
        param_group = self.param_groups[index]
        name_group = self.name_groups[index]
        handpose_group = torch.FloatTensor(self.handpose_groups[index])
        objpose_group = torch.FloatTensor(self.objpose_groups[index])
        proj_group = torch.FloatTensor(self.proj_groups[index])
        T_pose_21 = torch.FloatTensor(self.T_pose_21_groups[index])
        bone_length = torch.FloatTensor(self.bone_length_groups[index])
        obj_verts = self.obj_verts_groups[index]
        obj_faces = self.obj_faces_groups[index]
        hand_model_path = self.hand_model_path[index]
        obj_model_path = self.obj_model_path[index]
        save_base_path =self.save_base_path[index]
        test_param_group = self.test_param_groups[index]
        test_name_groups = self.test_name_groups[index]

        return image_group, mask_group, param_group, test_param_group, name_group,test_name_groups,\
               handpose_group, objpose_group, proj_group,\
               T_pose_21, bone_length,\
               obj_verts,\
               hand_model_path, obj_model_path,save_base_path,\
               index

    def __getitem__(self, index):
        return self.getitem(index)


class fit_video_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        fit_type,
        obj_name,
        frame_name,
    ):  
        view_name_list = ['21320018','21320027','21320028','21320029',\
                          '21320030','21320034','21320035','21320036',]

        self.img_groups = []
        self.mask_groups = []
        self.param_groups = []
        self.test_param_groups = []
        self.name_groups = []
        self.objpose_groups = []
        self.handpose_groups = []
        self.proj_groups = []
        self.fitparam_groups = []
        self.T_pose_21_groups = []
        self.bone_length_groups = []
        self.palm_angle_groups = []
        self.obj_verts_groups = []
        self.obj_faces_groups = []
        self.hand_model_path = []
        self.obj_model_path = []
        self.save_base_path = []
        self.test_name_groups = []
        frame_path = os.path.join(data_root, obj_name, frame_name)
        img_path = os.path.join(data_root,  obj_name, frame_name, 'MASK')
        per, obj = obj_name.split("_")
        mesh_file = os.path.join(frame_path, obj + '_ours.ply')
        obj_verts, obj_faces = load_ply(mesh_file)
        self.obj_verts = obj_verts / 1000.0
        ori_param_file = os.path.join(frame_path, per + '_tmppose.pickle')
        with open(ori_param_file,'rb') as f:
            ori_param = pickle.load(f)
        ori_T_pose_21 = ori_param['T_pose_21']
        self.T_pose_21 = ori_T_pose_21
        self.bone_length = get_bone_length(ori_T_pose_21)
        self.bone_length = self.bone_length
        self.T_pose_21 = self.T_pose_21
        self.hand_model_path = './exp/'+ per + '/wmask_realhand'
        self.obj_model_path = './exp/'+ obj + '/wmask_realobj'
        for frame_id in range(2000):
            cur_img_list = []
            cur_mask_list = []
            cur_param_list = []
            cur_img_name_list = []
            cur_objpose_list = []
            cur_handjoint_list = [] 
            cur_proj_list = []
            test_file = os.path.join(img_path, str(frame_id) + '_21320018.jpeg')
            if not os.path.exists(test_file):
                continue
            for view_name in view_name_list:
                file_name = str(frame_id) + '_' + view_name
                img_file = os.path.join(img_path, file_name + '.jpeg')
                param_file = os.path.join(frame_path, 'PARAM_266', file_name + '.pickle')
                cur_img_ori = cv2.imread(img_file)
                cur_img_ori = cv2.resize(cur_img_ori, (266,230))
                mask = (cur_img_ori > 10).all(axis=-1)[..., None].astype(np.uint8) * 255
                img = (np.array(cur_img_ori) / 255.).astype(np.float32)
                mask = (np.array(mask) / 255.).astype(np.float32)
                if fit_type == '123':
                    save_base_path = os.path.join('./fit_res/','view_' + str(len(view_name_list)), '123', obj_name, frame_name)
                elif fit_type == '1234':
                    save_base_path = os.path.join('./fit_res/', 'view_' +str(len(view_name_list)), '1234', obj_name, frame_name)                
                pred_pose_path = os.path.join('./fit_res/','view_' + str(len(view_name_list)), '12', obj_name, frame_name, 'pose_12')
                pred_pose_file = os.path.join(pred_pose_path, str(frame_id) + '.pickle')
                with open(pred_pose_file,'rb') as f:
                    pred_param = pickle.load(f)
                pred_joint3d = pred_param['pred_joint3d'].astype(np.float32)
                fit_1_Ro = pred_param['pred_Ro'].astype(np.float32)
                fit_1_To = pred_param['pred_To'].astype(np.float32)
                obj_pose = np.eye(4)
                obj_pose[:3,:3] = fit_1_Ro
                obj_pose[:3,3] = fit_1_To
                height = cur_img_ori.shape[0]
                width = cur_img_ori.shape[1]
                s = min(height, width)
                with open(param_file,'rb') as f:
                    param = pickle.load(f)
                R = param['cam_R'].T
                T = param['cam_T'] 
                fx_ndc = param['fx_ndc'] 
                fy_ndc = param['fy_ndc'] 
                px_ndc = param['px_ndc'] 
                py_ndc = param['py_ndc']
                K = np.eye(3)
                fx = -1.0 * fx_ndc * (s-1 ) / 2.0 
                fy = -1.0 * fy_ndc * (s-1 ) / 2.0
                cx = -1.0 * px_ndc * (s-1 ) / 2.0 + (width-1 ) / 2.0
                cy = -1.0 * py_ndc * (s-1 ) / 2.0 + (height-1 ) / 2.0
                K[0, 0], K[1, 1] = fx, fy
                K[0,2], K[1,2] = cx, cy
                view_trans = np.zeros((3, 4))
                view_trans[:3,:3] = R
                view_trans[:3,3] = T
                proj = K @ view_trans
                cur_img_list.append(img)
                cur_mask_list.append(mask)
                cur_img_name_list.append(file_name + '.jpeg')
                cur_handjoint_list.append(pred_joint3d)
                cur_objpose_list.append(obj_pose)
                cur_param_list.append(param)
                cur_proj_list.append(proj)

            if len(cur_param_list) != 0:
                self.img_groups.append(cur_img_list)
                self.mask_groups.append(cur_mask_list)
                self.param_groups.append(cur_param_list)
                self.name_groups.append(cur_img_name_list)
                self.handpose_groups.append(cur_handjoint_list)
                self.objpose_groups.append(cur_objpose_list)
                self.proj_groups.append(cur_proj_list)
                self.save_base_path.append(save_base_path)

        self.img_groups = np.array(self.img_groups)
        self.mask_groups = np.array(self.mask_groups)
        self.handpose_groups = np.array(self.handpose_groups)
        self.objpose_groups =np.array(self.objpose_groups)
        self.proj_groups = np.array(self.proj_groups)
        print(len(self.img_groups))

    def __len__(self):
        return len(self.img_groups)

    def getitem(self, index):
        image_group = torch.FloatTensor(self.img_groups[index])
        mask_group = torch.FloatTensor(self.mask_groups[index])
        param_group = self.param_groups[index]
        name_group = self.name_groups[index]
        handpose_group = torch.FloatTensor(self.handpose_groups[index])
        objpose_group = torch.FloatTensor(self.objpose_groups[index])
        proj_group = torch.FloatTensor(self.proj_groups[index])
        T_pose_21 = torch.FloatTensor(self.T_pose_21)
        bone_length = torch.FloatTensor(self.bone_length)
        obj_verts = self.obj_verts
        save_base_path =self.save_base_path[index]

        return image_group, mask_group, param_group, name_group,\
               handpose_group, objpose_group, proj_group,\
               T_pose_21, bone_length, \
               obj_verts,\
               save_base_path,self.hand_model_path, self.obj_model_path,\
               index

    def __getitem__(self, index):
        return self.getitem(index)

class get_res_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        view_num,
        fit_type,
        fit_id,
        get_render=False,
    ):  
        view_name_list = ['21320018']
        fit_id = fit_id
        if get_render == False:
            test_view_name_list = ['21320018']
        else:
            test_view_name_list = ['21320018','21320028','21320029',\
                                    '21320034','21320036',]
        ori_path = './data/catch_sequence/test'
        self.img_groups = []
        self.mask_groups = []
        self.param_groups = []
        self.test_param_groups = []
        self.name_groups = []
        self.objpose_groups = []
        self.handpose_groups = []
        self.proj_groups = []
        self.fitparam_groups = []
        self.T_pose_21_groups = []
        self.bone_length_groups = []
        self.palm_angle_groups = []
        self.obj_verts_groups = []
        self.obj_faces_groups = []
        self.hand_model_path = []
        self.obj_model_path = []
        self.save_base_path = []
        self.test_name_groups = []
        obj_list = os.listdir(data_root)
        obj_list.sort()
        for obj_name in obj_list:
            per, obj = obj_name.split("_")
            obj_path = os.path.join(data_root, obj_name)
            frame_list = os.listdir(obj_path)
            frame_list.sort()
            for frame_name in frame_list:
                frame_path = os.path.join(obj_path, frame_name)
                img_path = os.path.join(frame_path, 'MASK')
                mesh_file = os.path.join(frame_path, obj + '_ours.ply')
                obj_verts, obj_faces = load_ply(mesh_file)
                obj_verts /= 1000.0
                for frame_id in range(0,2000):
                    cur_img_list = []
                    cur_mask_list = []
                    cur_param_list = []
                    cur_img_name_list = []
                    cur_objpose_list = []
                    cur_handjoint_list = [] 
                    cur_proj_list = []
                    cur_fitparam_list = []
                    cur_test_param_list = []
                    cur_test_img_name_list = []
                    test_file = os.path.join(img_path, str(frame_id) + '_21320018.jpeg')
                    if not os.path.exists(test_file):
                        continue
                    person_name, object_name = obj_name.split('_')
                    hand_model_path = './exp/' + person_name + '/wmask_realhand'
                    obj_model_path = './exp/'+ object_name  + '/wmask_realobj'
                    for view_name in view_name_list:
                        file_name = str(frame_id) + '_' + view_name
                        param_file = os.path.join(frame_path, 'PARAM_266', file_name + '.pickle')
                        ori_param_file = os.path.join(frame_path,per + '_tmppose.pickle')
                        with open(ori_param_file,'rb') as f:
                            ori_param = pickle.load(f)
                        T_pose_21 = ori_param['T_pose_21']
                        bone_length = get_bone_length(T_pose_21)
                        with open(param_file,'rb') as f:
                            param = pickle.load(f)
                        if get_render == False:
                            if fit_type == '1' or fit_type == '12':
                                pose_type = fit_type
                            else:
                                pose_type = '4'
                            pred_pose_path = os.path.join('./fit_res', 'view_8', fit_type, obj_name, frame_name, 'pose_'+pose_type)
                            if not os.path.exists(pred_pose_path):
                                continue
                            pred_pose_file = os.path.join(pred_pose_path, str(frame_id) + '.pickle')
                            with open(pred_pose_file,'rb') as f:
                                pred_param = pickle.load(f)
                            pred_joint3d = pred_param['pred_joint3d'].astype(np.float32)
                            fit_1_Ro = pred_param['pred_Ro'].astype(np.float32)
                            fit_1_To = pred_param['pred_To'].astype(np.float32)
                            obj_pose = np.eye(4)
                            obj_pose[:3,:3] = fit_1_Ro
                            obj_pose[:3,3] = fit_1_To
                            save_base_path = os.path.join('./fit_res/analys_res', 'view_8', fit_type, obj_name, frame_name)
                        else:
                            if fit_type == '0':
                                pred_3djoint_file = os.path.join(ori_path,obj_name,frame_name, 'pred_joint3d_3view', str(frame_id) + '.pickle')
                                pred_objpose_file = os.path.join(ori_path,obj_name,frame_name, 'pred_objpose_3view', str(frame_id) + '.txt')
                                with open(pred_3djoint_file,'rb') as f:
                                    joint3d_param = pickle.load(f)
                                pred_joint3d = joint3d_param['pred_joint_3d'].astype(np.float32)
                                obj_pose = np.loadtxt(pred_objpose_file).astype(np.float32)
                            else:
                                pose_type = fit_type
                                pred_pose_path = os.path.join('./fit_res', 'view_3', fit_type, obj_name, frame_name, 'pose_'+pose_type)
                                pred_pose_file = os.path.join(pred_pose_path, str(frame_id) + '.pickle')
                                with open(pred_pose_file,'rb') as f:
                                    pred_param = pickle.load(f)
                                pred_joint3d = pred_param['pred_joint3d'].astype(np.float32)
                                fit_1_Ro = pred_param['pred_Ro'].astype(np.float32)
                                fit_1_To = pred_param['pred_To'].astype(np.float32)
                                obj_pose = np.eye(4)
                                obj_pose[:3,:3] = fit_1_Ro
                                obj_pose[:3,3] = fit_1_To
                            save_base_path = os.path.join('./fit_res/analys_res', 'view_3', fit_type, obj_name, frame_name)

                        height = param['H']
                        width = param['W']
                        s = min(height, width)
                        R = param['cam_R'].T
                        T = param['cam_T'] 
                        fx_ndc = param['fx_ndc'] 
                        fy_ndc = param['fy_ndc'] 
                        px_ndc = param['px_ndc'] 
                        py_ndc = param['py_ndc']
                        K = np.eye(3)
                        fx = -1.0 * fx_ndc * (s-1 ) / 2.0 
                        fy = -1.0 * fy_ndc * (s-1 ) / 2.0
                        cx = -1.0 * px_ndc * (s-1 ) / 2.0 + (width-1 ) / 2.0
                        cy = -1.0 * py_ndc * (s-1 ) / 2.0 + (height-1 ) / 2.0
                        K[0, 0], K[1, 1] = fx, fy
                        K[0,2], K[1,2] = cx, cy
                        view_trans = np.zeros((3, 4))
                        view_trans[:3,:3] = R
                        view_trans[:3,3] = T
                        proj = K @ view_trans

                        cur_img_name_list.append(file_name + '.jpeg')
                        cur_handjoint_list.append(pred_joint3d)
                        cur_objpose_list.append(obj_pose)
                        cur_param_list.append(param)
                        cur_proj_list.append(proj)
                    for view_name in test_view_name_list:
                        file_name = str(frame_id) + '_' + view_name
                        param_file = os.path.join(frame_path, 'PARAM_266', file_name + '.pickle')
                        with open(param_file,'rb') as f:
                            param = pickle.load(f)
                        cur_test_param_list.append(param)
                        cur_test_img_name_list.append(file_name + '.jpeg')
                        
                    if len(cur_param_list) != 0:
                        self.param_groups.append(cur_param_list)
                        self.name_groups.append(cur_img_name_list)
                        self.handpose_groups.append(cur_handjoint_list)
                        self.objpose_groups.append(cur_objpose_list)
                        self.proj_groups.append(cur_proj_list)
                        self.T_pose_21_groups.append(T_pose_21)
                        self.bone_length_groups.append(bone_length)
                        self.obj_verts_groups.append(obj_verts)
                        self.obj_faces_groups.append(obj_faces)
                        self.hand_model_path.append(hand_model_path)
                        self.obj_model_path.append(obj_model_path)
                        self.save_base_path.append(save_base_path)
                        self.test_param_groups.append(cur_test_param_list)
                        self.test_name_groups.append(cur_test_img_name_list)

        self.handpose_groups = np.array(self.handpose_groups)
        self.objpose_groups = np.array(self.objpose_groups)
        self.proj_groups = np.array(self.proj_groups)
        self.T_pose_21_groups = np.array(self.T_pose_21_groups)
        self.bone_length_groups = np.array(self.bone_length_groups)
        self.obj_verts_groups = self.obj_verts_groups
        print(self.handpose_groups.shape[0])

    def __len__(self):
        return self.handpose_groups.shape[0]

    def getitem(self, index):

        param_group = self.param_groups[index]
        name_group = self.name_groups[index]
        handpose_group = torch.FloatTensor(self.handpose_groups[index])
        objpose_group = torch.FloatTensor(self.objpose_groups[index])
        proj_group = torch.FloatTensor(self.proj_groups[index])
        T_pose_21 = torch.FloatTensor(self.T_pose_21_groups[index])
        bone_length = torch.FloatTensor(self.bone_length_groups[index])
        obj_verts = self.obj_verts_groups[index]
        obj_faces = self.obj_faces_groups[index]
        hand_model_path = self.hand_model_path[index]
        obj_model_path = self.obj_model_path[index]
        save_base_path =self.save_base_path[index]
        test_param_group = self.test_param_groups[index]
        test_name_groups = self.test_name_groups[index]

        return param_group, test_param_group, name_group,test_name_groups,\
               handpose_group, objpose_group, proj_group,\
               T_pose_21, bone_length,\
               obj_verts,\
               hand_model_path, obj_model_path,save_base_path,\
               index

    def __getitem__(self, index):
        return self.getitem(index)
    