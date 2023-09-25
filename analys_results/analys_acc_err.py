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
import open3d as o3d

def load_ply(pth):
    mesh = o3d.io.read_triangle_mesh(pth)
    vts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    return vts, faces

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2)  /sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def get_acc_list(pose_path_1_3, pose_path_123, pose_path_1234, obj_v, st_id, end_id):
    j_list_1_3 = []
    j_list_123 = []
    j_list_1234 = []
    j_list_gt = []

    v_list_1_3 = []
    v_list_123 = []
    v_list_1234 = []
    v_list_gt = []
    
    for cid in range(st_id, end_id):
        file_name = str(cid) + '.pickle'
        pose_file_1_3 = os.path.join(pose_path_1_3, file_name)
        if not os.path.exists(pose_file_1_3):
            continue
        with open(pose_file_1_3,'rb') as f:
            param_1_3 = pickle.load(f)
        joint3d_1_3 = param_1_3['pred_joint3d']
        Ro_1_3 = param_1_3['pred_Ro']
        To_1_3 = param_1_3['pred_To']
        obj_v_1_3 = obj_v @ Ro_1_3.T + To_1_3
        pose_file_123 = os.path.join(pose_path_123, file_name)
        with open(pose_file_123,'rb') as f:
            param_123 = pickle.load(f)
        joint3d_123 = param_123['pred_joint3d']
        Ro_123 = param_123['pred_Ro']
        To_123 = param_123['pred_To']
        obj_v_123 = obj_v @ Ro_123.T + To_123

        gt_joint3d = param_123['gt_joint3d']
        gt_Ro = param_123['gt_Ro']
        gt_To = param_123['gt_To']
        obj_v_gt = obj_v @ gt_Ro.T + gt_To

        pose_file_1234 = os.path.join(pose_path_1234, file_name)
        with open(pose_file_1234,'rb') as f:
            param_1234 = pickle.load(f)
        joint3d_1234 = param_1234['pred_joint3d']
        Ro_1234 = param_1234['pred_Ro']
        To_1234 = param_1234['pred_To']
        obj_v_1234 = obj_v @ Ro_1234.T + To_1234

        j_list_1_3.append(joint3d_1_3)
        j_list_123.append(joint3d_123)
        j_list_1234.append(joint3d_1234)
        j_list_gt.append(gt_joint3d)

        v_list_1_3.append(obj_v_1_3)
        v_list_123.append(obj_v_123)
        v_list_1234.append(obj_v_1234)
        v_list_gt.append(obj_v_gt)

    j_list_1_3 = np.array(j_list_1_3)
    j_list_123 = np.array(j_list_123)
    j_list_1234 = np.array(j_list_1234)
    j_list_gt = np.array(j_list_gt)

    v_list_1_3 = np.array(v_list_1_3)
    v_list_123 = np.array(v_list_123)
    v_list_1234 = np.array(v_list_1234)
    v_list_gt = np.array(v_list_gt)
    
    acc_1_3_j = compute_error_accel(j_list_gt, j_list_1_3)
    acc_123_j = compute_error_accel(j_list_gt, j_list_123)
    acc_1234_j = compute_error_accel(j_list_gt, j_list_1234)

    acc_1_3_v = compute_error_accel(v_list_gt, v_list_1_3)
    acc_123_v = compute_error_accel(v_list_gt, v_list_123)
    acc_1234_v = compute_error_accel(v_list_gt, v_list_1234)
    cnum = v_list_1_3.shape[0]
    return acc_1_3_j, acc_123_j, acc_1234_j,acc_1_3_v, acc_123_v, acc_1234_v, cnum

if __name__ == '__main__':

    base_path = './fit_res/view_8' # /combine
    joint3d_err_ours = 0
    joint3d_err_init = 0    
    model_base_path = './data/offline_stage_data'

    acc_list_1_3_j = []
    acc_list_123_j = []
    acc_list_1234_j = []
    acc_list_1_3_v = []
    acc_list_123_v = []
    acc_list_1234_v = []

    num_all = 0
    sub_path = os.path.join(base_path, '1234')
    
    for obj_name in os.listdir(sub_path):
        obj_path = os.path.join(sub_path, obj_name)
        per, obj = obj_name.split('_')
        obj_model_file = os.path.join(model_base_path, obj + '_cppose', obj + '_ours.ply')
        obj_verts, obj_faces = load_ply(obj_model_file)
        obj_verts = obj_verts / 1000.0
        
        for frame_name in os.listdir(obj_path):
            frame_path = os.path.join(obj_path, frame_name)
            pose_path_1_3 = os.path.join(base_path, '12', obj_name, frame_name, 'pose_12')
            pose_path_123 = os.path.join(base_path, '123', obj_name, frame_name, 'pose_4')
            pose_path_1234 = os.path.join(base_path, '1234', obj_name, frame_name, 'pose_4')
            acc_1_3_j,acc_123_j,acc_1234_j,acc_1_3_v, acc_123_v, acc_1234_v, cnum = get_acc_list(pose_path_1_3, pose_path_123, pose_path_1234,obj_verts, 0, 2000)
            acc_list_1_3_j.append(acc_1_3_j)
            acc_list_123_j.append(acc_123_j)
            acc_list_1234_j.append(acc_1234_j)
            acc_list_1_3_v.append(acc_1_3_v)
            acc_list_123_v.append(acc_123_v)
            acc_list_1234_v.append(acc_1234_v)
            num_all += cnum

    acc_list_1_3_j = np.concatenate(acc_list_1_3_j)
    acc_list_1_3_j = acc_list_1_3_j.mean() * 1000.0
    acc_list_123_j = np.concatenate(acc_list_123_j)
    acc_list_123_j = acc_list_123_j.mean() * 1000.0
    acc_list_1234_j = np.concatenate(acc_list_1234_j)
    acc_list_1234_j = acc_list_1234_j.mean() * 1000.0

    acc_list_1_3_v = np.concatenate(acc_list_1_3_v)
    acc_list_1_3_v = acc_list_1_3_v.mean() * 1000.0
    acc_list_123_v = np.concatenate(acc_list_123_v)
    acc_list_123_v = acc_list_123_v.mean() * 1000.0
    acc_list_1234_v = np.concatenate(acc_list_1234_v)
    acc_list_1234_v = acc_list_1234_v.mean() * 1000.0

    print(num_all)
    print('acc_list_1_3_j: %.2lf, acc_list_1_3_v: %.2lf, acc_list_123_j: %.2lf,  acc_list_123_v: %.2lf,  acc_list_1234_j: %.2lf, acc_list_1234_v: %.2lf'\
                     %(acc_list_1_3_j,acc_list_1_3_v, acc_list_123_j,acc_list_123_v, acc_list_1234_j,acc_list_1234_v))

        





