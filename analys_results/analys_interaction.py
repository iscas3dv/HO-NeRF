import sys
import os
import time
import argparse
import pickle
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import tqdm
import trimesh
import matplotlib.pyplot as plt

def intersect_vox(obj_mesh, hand_mesh, pitch=0.005):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def get_int_vol(mesh_path_hand, mesh_path_obj, int_file):
    if os.path.exists(int_file):
        print(int_file)
        with open(int_file,'rb') as f:
            int_param = pickle.load(f)
        int_vol = int_param['int_vol']
        pen_dep = int_param['pen_dep']
    else:
        hand_mesh = trimesh.load(mesh_path_hand)
        hand_mesh.vertices = hand_mesh.vertices
        obj_mesh = trimesh.load(mesh_path_obj)
        obj_mesh.vertices = obj_mesh.vertices
        int_vol = intersect_vox(obj_mesh, hand_mesh)
        int_vol *= 1e6
        pen_dep = get_pen_depth(mesh_path_hand,mesh_path_obj)
        param = {}
        param['int_vol'] = int_vol
        param['pen_dep'] = pen_dep
        f = open(int_file,'wb')
        pickle.dump(param,f)
        f.close()
    return int_vol, pen_dep

def get_pen_depth(mesh_path_hand, mesh_path_obj):
    hand_mesh = trimesh.load(mesh_path_hand)
    obj_mesh = trimesh.load(mesh_path_obj)
    hand_points = hand_mesh.vertices
    obj_points = obj_mesh.vertices
    init_id = obj_mesh.contains(hand_points)
    if init_id.sum() == 0:
        return 0
    result_close,result_distance,_ = trimesh.proximity.closest_point(obj_mesh, hand_points[init_id])
    max_depth = result_distance.max() * 1000.0
    
    return max_depth

if __name__ == '__main__':  

    base_path = './fit_res/analys_res/view_8' # /combine
    class_type = ['bean', 'box', 'cup', 'meat']

    for cur_class in class_type:
        first_int_sum = 0
        first_dep_sum = 0
        second_int_sum = 0
        second_dep_sum = 0
        cid = 0
        sub_path = os.path.join(base_path, '1')
        for obj_name in os.listdir(sub_path):
            if cur_class not in obj_name:
                continue
            obj_path = os.path.join(sub_path, obj_name)

            for frame_name in os.listdir(obj_path):
                frame_path = os.path.join(obj_path, frame_name)
                print(obj_name, frame_name)
                for frame_id in range(2000):
                    first_hand = os.path.join(frame_path, 'mesh_1', '{:d}_hand.ply'.format(frame_id))
                    first_obj = os.path.join(frame_path, 'mesh_1', '{:d}_obj.ply'.format(frame_id))
                    first_int_path = os.path.join(frame_path,'int')
                    os.makedirs(first_int_path, exist_ok=True)
                    first_int_file = os.path.join(first_int_path,'{:d}.pickle'.format(frame_id))
                    if not os.path.exists(first_hand):
                        continue
                    second_hand = os.path.join(base_path, '12', obj_name, frame_name, 'mesh_12','{:d}_hand.ply'.format(frame_id))
                    second_obj = os.path.join(base_path, '12', obj_name, frame_name, 'mesh_12','{:d}_obj.ply'.format(frame_id))
                    second_int_path = os.path.join(base_path, '12', obj_name, frame_name, 'int')
                    os.makedirs(second_int_path, exist_ok=True)
                    second_int_file = os.path.join(second_int_path,'{:d}.pickle'.format(frame_id))
                    first_int, first_dep = get_int_vol(first_hand, first_obj, first_int_file)
                    second_int, second_dep = get_int_vol(second_hand, second_obj, second_int_file)
                    first_int_sum += first_int
                    first_dep_sum += first_dep
                    second_int_sum += second_int
                    second_dep_sum += second_dep
                    cid += 1

        first_int_sum /= cid
        first_dep_sum /= cid
        second_int_sum /= cid
        second_dep_sum /= cid
        
        print('object class %s has %d frames' % (cur_class, cid))
        print('fit1_int_sum: %.2lf, fit1_dep_sum: %.2lf, fit12_int_sum: %.2lf, fit12_dep_sum: %.2lf'\
                        %(first_int_sum, first_dep_sum, second_int_sum, second_dep_sum))
