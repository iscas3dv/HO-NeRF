import os
import pickle
import torch
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image
import pandas as pd
import numpy as np
import tqdm
import cv2
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
import math
import scipy.linalg as linalg

def load_ply(pth):
    mesh = o3d.io.read_triangle_mesh(pth)
    vts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    return vts, faces

def get_inner_point_id_from_file(file):
    with open(file,'rb') as f:
        param = pickle.load(f)
    inner_point_id = param['inner_point_id']
    return inner_point_id

def get_iou_map(file_path, cid, gap):

    pre_file = os.path.join(file_path, '{:d}.pickle'.format(cid-gap))
    pre_id = get_inner_point_id_from_file(pre_file)
    next_file = os.path.join(file_path, '{:d}.pickle'.format(cid))
    next_id = get_inner_point_id_from_file(next_file)

    union = np.union1d(pre_id, next_id)
    inter = np.intersect1d(pre_id, next_id)

    inter_num = inter.shape[0]
    union_num = union.shape[0] + 1e-7
    return inter_num / union_num

if __name__ == '__main__':  

    base_path = './fit_res/analys_res/view_8' # /combine
    model_base_path = './data/offline_stage_data'
    cid = 0
    first_int_list = []
    second_int_list = []
    third_int_list = []

    first_sum = 0
    second_sum = 0
    third_sum = 0

    sub_path = os.path.join(base_path, '1234')
    for obj_name in os.listdir(sub_path):
        obj_path = os.path.join(sub_path, obj_name)
        per, obj = obj_name.split('_')
        obj_model_file = os.path.join(model_base_path, obj + '_cppose', obj + '_ours.ply')
        obj_verts, obj_faces = load_ply(obj_model_file)
        obj_verts = obj_verts / 1000.0
        for frame_name in os.listdir(obj_path):
            frame_path = os.path.join(obj_path, frame_name)
            print(obj_name, frame_name)
            vert_count_1 = np.zeros(obj_verts.shape[0])
            vert_count_2 = np.zeros(obj_verts.shape[0])
            vert_count_3 = np.zeros(obj_verts.shape[0])
            gap = 1
            first_root = os.path.join(base_path, '12', obj_name, frame_name, 'inner_12')
            second_root = os.path.join(base_path, '123', obj_name, frame_name, 'inner_123')
            third_root = os.path.join(base_path, '1234', obj_name, frame_name, 'inner_1234')
            start_flag = 0
            for frame_id in range(2000):
                first_inner_file = os.path.join(first_root,'{:d}.pickle'.format(frame_id))
                if not os.path.exists(first_inner_file):
                    continue
                if start_flag == 0:
                    start_flag = 1
                    continue
                cur_id = frame_id
                first_iou = get_iou_map(first_root, frame_id,gap)
                second_iou = get_iou_map(second_root, frame_id,gap)
                third_iou = get_iou_map(third_root, frame_id,gap)
                first_sum += first_iou
                second_sum += second_iou
                third_sum += third_iou
                cid += 1

    print('inner_12 pci: %.2lf, inner_123 pci: %.2lf, inner_124 pci: %.2lf' % (first_sum / cid * 100 , second_sum / cid * 100 , third_sum / cid * 100))
    print(cid)
    