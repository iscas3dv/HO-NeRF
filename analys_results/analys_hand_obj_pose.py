import os
from typing import List, Optional, Tuple
import imageio 
import numpy as np
import requests
import torch
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
import cv2
from scipy import spatial
import open3d as o3d

def add(pred_pts, gt_pts):
    e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return e

def adi(pred_pts, gt_pts):
    nn_index = spatial.cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1)
    e = nn_dists.mean()
    return e

def load_ply(pth):
    mesh = o3d.io.read_triangle_mesh(pth)
    vts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    return vts, faces

if __name__ == '__main__':

    base_path = './fit_res'
    obj_list = ['bean', 'box', 'cup', 'meat']
    fit_type = '12' # '1', '12', '123', '1234'
    view_num = '8' # '8', '6', '3'
    model_base_path = './data/offline_stage_data'
    init_path = './data/catch_sequence/test'

    for test_obj in obj_list:
        joint3d_err_ours = 0
        joint3d_err_init = 0
        objpose_err_ours = 0
        objpose_err_init = 0
        add_err_ours = 0
        add_err_init = 0
        adi_err_ours = 0
        adi_err_init = 0
        thread = 0.015
        cnum = 0
        sub_name_view = 'view_' + view_num
        sub_path = os.path.join(base_path, sub_name_view)
        type_path = os.path.join(sub_path, fit_type)
        for obj_name in os.listdir(type_path):
            if test_obj not in obj_name:
                continue
            per, obj = obj_name.split('_')
            obj_model_file = os.path.join(model_base_path, obj + '_cppose', obj + '_ours.ply')
            vert_model,_ = load_ply(obj_model_file)
            vert_model /= 1000.0

            obj_path = os.path.join(type_path, obj_name)
            for frame_name in os.listdir(obj_path):
                f_id = 0
                frame_path = os.path.join(obj_path, frame_name)
                pose_path = os.path.join(frame_path, 'pose_' + fit_type)

                init_jonit3d_path = os.path.join(init_path, obj_name,frame_name, 'pred_joint3d_' + view_num + 'view')
                init_objpose_path = os.path.join(init_path, obj_name,frame_name, 'pred_objpose_' + view_num+ 'view')

                for file_name in os.listdir(pose_path):
                    pose_file = os.path.join(pose_path, file_name)
                    cid = file_name.split('.')[0]

                    init_joint_file = os.path.join(init_jonit3d_path, cid + '.pickle')
                    init_objpose_file = os.path.join(init_objpose_path, cid + '.txt')
                    with open(pose_file,'rb') as f:
                        param = pickle.load(f)

                    ours_joint3d = param['pred_joint3d']
                    ours_Ro = param['pred_Ro']
                    ours_To = param['pred_To']

                    gt_joint3d = param['gt_joint3d']
                    gt_Ro = param['gt_Ro']
                    gt_To = param['gt_To']

                    with open(init_joint_file,'rb') as f:
                        joint3d_param = pickle.load(f)
                    init_joint3d = joint3d_param['pred_joint_3d'].astype(np.float32)
                    init_obj_pose = np.loadtxt(init_objpose_file).astype(np.float32)
                    init_Ro = init_obj_pose[:3,:3]
                    init_To = init_obj_pose[:3,3]

                    cur_ours_joint_err = (np.sqrt(((ours_joint3d - gt_joint3d)**2).sum(1))).mean()
                    joint3d_err_ours += cur_ours_joint_err
                    cur_pred_joint_err = (np.sqrt(((init_joint3d - gt_joint3d)**2).sum(1))).mean()
                    joint3d_err_init += cur_pred_joint_err

                    cur_ours_obj_v = vert_model @ ours_Ro.T + ours_To
                    cur_init_obj_v = vert_model @ init_Ro.T + init_To
                    cur_gt_obj_v = vert_model @ gt_Ro.T + gt_To

                    cur_ours_obj_err = (np.sqrt(((cur_ours_obj_v - cur_gt_obj_v)**2).sum(1))).mean()
                    objpose_err_ours += cur_ours_obj_err
                    cur_init_obj_err = (np.sqrt(((cur_init_obj_v - cur_gt_obj_v)**2).sum(1))).mean()
                    objpose_err_init += cur_init_obj_err

                    ours_add = add(cur_ours_obj_v, cur_gt_obj_v)
                    if ours_add < thread:
                        add_err_ours += 1
                    ours_adi = adi(cur_ours_obj_v, cur_gt_obj_v)
                    if ours_adi < thread:
                        adi_err_ours += 1
                    init_add = add(cur_init_obj_v, cur_gt_obj_v)
                    if init_add < thread:
                        add_err_init += 1
                    init_adi = adi(cur_init_obj_v, cur_gt_obj_v)
                    if init_adi < thread:
                        adi_err_init += 1

                    cnum += 1
                    f_id +=1
                print('obj_name: %s,  frame_name: %s, cur_frame_num: %d' %(obj_name, frame_name,f_id))

        print('obj_name %s has %d frames' % (test_obj, cnum))
        joint3d_err_ours /= cnum
        joint3d_err_ours *= 1000
        joint3d_err_init /= cnum
        joint3d_err_init *= 1000

        objpose_err_ours /= cnum
        objpose_err_ours *= 1000
        objpose_err_init /= cnum
        objpose_err_init *= 1000

        add_err_ours /= cnum
        add_err_ours *= 100
        adi_err_ours /= cnum
        adi_err_ours *= 100

        add_err_init /= cnum
        add_err_init *= 100
        adi_err_init /= cnum
        adi_err_init *= 100

        print('init joint: %.2lf, ours joint: %.2lf, init ad: %.2lf, init add: %.2lf, init adds: %.2lf, ours ad: %.2lf, ours add: %.2lf, ours adds: %.2lf'\
                            %(joint3d_err_init, joint3d_err_ours, objpose_err_init,add_err_init,adi_err_init, objpose_err_ours, add_err_ours, adi_err_ours))

