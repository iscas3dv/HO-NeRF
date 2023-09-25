import os
import cv2
import time
import logging
import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import shutil
import trimesh
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from utils.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, RenderingNetwork_OBJ, SDFNetwork_OBJ, Embedding, VGGLoss
from utils.renderer import NeuSRenderer_fitting
from utils.utils import  rot6d_to_matrix, _xy_to_ray_bundle
from utils.dataset import get_res_dataset, get_rays_xy
from pytorch3d.renderer import PerspectiveCameras
from halo_util.utils import convert_joints
from halo_util.converter_fit_batch import PoseConverter, transform_to_canonical


class Runner:
    def __init__(self, conf_path, case='CASE_NAME', render=False):
        self.conf_path = conf_path
        self.case = case
        self.render = render
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.device = torch.device('cuda')
        self.base_exp_dir = self.conf['general.save_dir']
        self.save_dir = self.conf['general.save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_type = self.conf.get_string('general.model_type')
        self.data_root = self.conf.get_string('dataset.fitdata_dir')
        self.data_type = self.conf.get_string('general.data_type')
        self.fit_type = self.conf.get_string('general.fit_type')
        self.fit_id = self.conf.get_int('general.fit_id')
        self.view_num = self.conf.get_string('dataset.view_num')
        self.H, self.W = self.conf.get_list('dataset.image_size')
        fit_dataset = get_res_dataset(
            data_root = self.data_root,
            view_num = self.view_num,
            fit_type = self.fit_type,
            fit_id = self.fit_id,
            get_render = self.render,
            )
        self.fit_dataloader = torch.utils.data.DataLoader(
            fit_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator(device='cuda')
            )
        self.batch_size = self.conf.get_int('train.batch_size')
        self.near = self.conf['train.near']
        self.far = self.conf['train.far']
        self.model_list = []
        self.writer = None

    def get_latest_name(self, base_exp_dir):
        model_list_raw = os.listdir(os.path.join(base_exp_dir, 'checkpoints'))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth':
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        return latest_model_name

    def file_backup(self,base_exp_dir):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.save_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.save_dir,'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.save_dir, 'recording', 'config.conf'))

    def get_bound(self, verts):
        cur_verts = verts.cpu().detach().numpy()
        padding = 0.08
        x_min = cur_verts[:,0].min() - padding
        x_max = cur_verts[:,0].max() + padding
        y_min = cur_verts[:,1].min() - padding
        y_max = cur_verts[:,1].max() + padding
        z_min = cur_verts[:,2].min() - padding
        z_max = cur_verts[:,2].max() + padding
        bound_min = np.array([x_min, y_min, z_min])
        bound_max = np.array([x_max, y_max, z_max])
        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)
        return bound_min, bound_max

    def init_model(self, hand_model_dir, obj_model_dir):
        
        self.barf_encoding = Embedding().to(self.device)
        self.pose_converter = PoseConverter(dev=self.device)
        self.deviation_network_hand = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        hand_latest_model_name = self.get_latest_name(hand_model_dir)
        hand_checkpoint = torch.load(os.path.join(hand_model_dir, 'checkpoints', hand_latest_model_name), map_location=self.device)
        if self.data_type == 'real':
            hand_barf_len = hand_checkpoint['sdf_network_fine']['se3_refine'].shape[0]
        else:
            hand_barf_len = 0
        self.sdf_network_hand = SDFNetwork(self.barf_encoding,hand_barf_len,self.data_type,**self.conf['model.sdf_hand_network']).to(self.device)
        self.color_network_hand = RenderingNetwork(self.barf_encoding,self.data_type,**self.conf['model.rendering_hand_network']).to(self.device)
        self.sdf_network_hand.load_state_dict(hand_checkpoint['sdf_network_fine'],strict=False)
        self.deviation_network_hand.load_state_dict(hand_checkpoint['variance_network_fine'],strict=False)
        self.color_network_hand.load_state_dict(hand_checkpoint['color_network_fine'],strict=False)
        self.deviation_network_obj = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        obj_latest_model_name = self.get_latest_name(obj_model_dir)
        obj_checkpoint = torch.load(os.path.join(obj_model_dir, 'checkpoints', obj_latest_model_name), map_location=self.device)
        if self.data_type == 'real':
            obj_barf_len = obj_checkpoint['sdf_network_fine']['se3_refine'].shape[0]
        else:
            obj_barf_len = 0
        self.sdf_network_obj = SDFNetwork_OBJ(self.barf_encoding,obj_barf_len,self.data_type,**self.conf['model.sdf_obj_network']).to(self.device)
        self.color_network_obj = RenderingNetwork_OBJ(self.barf_encoding,self.data_type,**self.conf['model.rendering_obj_network']).to(self.device)
        self.sdf_network_obj.load_state_dict(obj_checkpoint['sdf_network_fine'],strict=False)
        self.deviation_network_obj.load_state_dict(obj_checkpoint['variance_network_fine'],strict=False)
        self.color_network_obj.load_state_dict(obj_checkpoint['color_network_fine'],strict=False)
        self.renderer = NeuSRenderer_fitting(self.sdf_network_hand,
                                     self.deviation_network_hand,
                                     self.color_network_hand,
                                     self.sdf_network_obj,
                                     self.deviation_network_obj,
                                     self.color_network_obj,
                                     **self.conf['model.neus_renderer'])

    def fitting(self):

        for batch_idx, test_batch in enumerate(self.fit_dataloader):
            param_group,test_param_group,name_group,test_name_groups,\
            pred_joint_group, pred_objpose_group, proj_group, \
            T_pose_21, bone_length,\
            obj_verts,\
            hand_model_path, obj_model_path,save_base_path,\
            index = test_batch
            save_base_path = save_base_path[0]
            self.init_model(hand_model_path[0], obj_model_path[0])
            get_mesh = False
            get_inner_p = False
            get_render = False
            if self.render:
                save_render_path = os.path.join(save_base_path, 'render_' + self.fit_type)
                os.makedirs(save_render_path, exist_ok=True)
                get_render = True
            else:
                if self.fit_type == '1' or self.fit_type == '12':
                    mesh_path = os.path.join(save_base_path, 'mesh_' + self.fit_type)
                    os.makedirs(mesh_path, exist_ok=True)
                    get_mesh = True
                if self.fit_type == '12' or self.fit_type == '123' or self.fit_type == '1234':
                    save_inner_path = os.path.join(save_base_path, 'inner_' + self.fit_type)
                    os.makedirs(save_inner_path, exist_ok=True)
                    get_inner_p = True
            cid_name = name_group[0][0].split('_')[0]
            Ro_pred = pred_objpose_group[0,0,:3,:3].to(self.device).float()
            To_pred = pred_objpose_group[0,0,:3,3].to(self.device).float()
            joint3d_pred = pred_joint_group[0,0].to(self.device).float()
            T_pose_21 = T_pose_21.to(self.device)[0].float()
            cur_bone_length = bone_length.to(self.device).float()
            obj_verts = obj_verts.to(self.device)[0].float()
            ori_3d_pose = joint3d_pred.unsqueeze(0)
            ori_obj_r = Ro_pred
            ori_obj_t = To_pred
            obj_rot_refine = torch.eye(3).to(self.device)
            obj_rot_refine = obj_rot_refine[:,:2]
            obj_rot_refine = torch.nn.Parameter(obj_rot_refine.float(), requires_grad=False)
            obj_trans_refine = torch.zeros(3).to(self.device)
            obj_trans_refine = torch.nn.Parameter(obj_trans_refine.float(), requires_grad=False)
            palm_rot_refine = torch.eye(3).to(self.device)
            palm_rot_refine = palm_rot_refine[:,:2]
            palm_rot_refine = torch.nn.Parameter(palm_rot_refine.unsqueeze(0).float(), requires_grad=False)
            palm_trans_refine = torch.zeros(3).to(self.device)
            palm_trans_refine = torch.nn.Parameter(palm_trans_refine.unsqueeze(0).float(), requires_grad=False)
            joint_refine_angle = torch.zeros(20).to(self.device)
            joint_refine_angle = torch.nn.Parameter(joint_refine_angle.unsqueeze(0).float(), requires_grad=False)
            palm_refine_angle = torch.zeros(7).to(self.device)
            palm_refine_angle = torch.nn.Parameter(palm_refine_angle.unsqueeze(0).float(), requires_grad=False)
            view_id = 0
            camera = PerspectiveCameras(R = param_group[view_id]['cam_R'], T = param_group[view_id]['cam_T'],\
                    focal_length = ((param_group[view_id]['fx_ndc'],param_group[view_id]['fy_ndc']),),\
                    principal_point = ((param_group[view_id]['px_ndc'],param_group[view_id]['py_ndc']),)).to(self.device)
            kps_local_cs = convert_joints(ori_3d_pose, source='mano', target='biomech')
            is_right_one = torch.ones(ori_3d_pose.shape[0], device=kps_local_cs.device)
            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
            joint_3d = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length,
                            joint_refine_angle=joint_refine_angle, palm_refine_angle=palm_refine_angle*0.1)
            glo_rot_right_inv = torch.inverse(glo_rot_right)
            joint_3d = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ joint_3d.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
            hand_rots = rot6d_to_matrix(palm_rot_refine)
            R_palm = hand_rots
            T_palm = palm_trans_refine
            joint_3d_root = joint_3d[:,:1,:].clone()
            joint_3d = (R_palm.unsqueeze(1) @ (joint_3d - joint_3d_root).unsqueeze(-1))[...,0] + joint_3d_root + T_palm.unsqueeze(1)
            kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech').to(self.device)
            is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
            rot_then_swap_mat = glo_rot_right.unsqueeze(1)
            trans_mat_pc, to_tpalm_angle,to_tpose_angle = self.pose_converter(palm_align_kps_local_cs, is_right_one)
            trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
            trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
            trans_mat_pc = trans_mat_pc_all
            bone_transformation_inv = trans_mat_pc[0]
            obj_rots = rot6d_to_matrix(obj_rot_refine)[0]
            obj_trans = obj_trans_refine
            obj_r = obj_rots @ ori_obj_r
            obj_t = ori_obj_t + obj_trans
            cur_obj_verts = (torch.matmul(obj_r.unsqueeze(0), obj_verts.unsqueeze(-1))[...,0] + obj_t.unsqueeze(0))
            if get_mesh:
                resolution = 64
                bound_min_hand, bound_max_hand = self.get_bound(joint_3d[0])
                verts_hand, faces_hand = self.renderer.extract_geometry(bound_min_hand, bound_max_hand, resolution, 
                                                    bone_transformation_inv, T_pose_21,
                                                    obj_r.T,obj_t, 'hand',threshold=0)
                bound_min_obj, bound_max_obj = self.get_bound(cur_obj_verts)
                verts_obj, faces_obj = self.renderer.extract_geometry(bound_min_obj, bound_max_obj, resolution, 
                                                    bone_transformation_inv, T_pose_21,
                                                    obj_r.T,obj_t, 'obj',threshold=0)
                mesh_hand = trimesh.Trimesh(verts_hand, faces_hand)
                mesh_file_hand = os.path.join(mesh_path, '{}_hand.ply'.format(cid_name))
                mesh_hand.export(mesh_file_hand)
                mesh_obj = trimesh.Trimesh(verts_obj, faces_obj)
                mesh_file_obj = os.path.join(mesh_path, '{}_obj.ply'.format(cid_name))
                mesh_obj.export(mesh_file_obj)
                print(mesh_file_hand)
            if get_inner_p:
                inner_point_id_list = self.renderer.get_inner_point_id(cur_obj_verts, bone_transformation_inv, T_pose_21)  
                param = {}
                param['inner_point_id'] = inner_point_id_list
                save_file_name = cid_name + '.pickle'
                save_file = os.path.join(save_inner_path, save_file_name)
                f = open(save_file,'wb')
                pickle.dump(param,f)
                f.close()
                print(save_file)
            if get_render:
                for i in range(len(test_param_group)):
                    R = test_param_group[i]['cam_R'].to(self.device)
                    T = test_param_group[i]['cam_T'].to(self.device)
                    camera = PerspectiveCameras(R = R, T = T,\
                            focal_length = ((test_param_group[i]['fx_ndc'],test_param_group[i]['fy_ndc']),),\
                            principal_point = ((test_param_group[i]['px_ndc'],test_param_group[i]['py_ndc']),)).to(self.device)
                    if self.W >= self.H:
                        range_x = self.W / self.H
                        range_y = 1.0
                    else:
                        range_x = 1.0
                        range_y = self.H / self.W
                    min_x = range_x 
                    max_x = -range_x 
                    min_y = range_y 
                    max_y = -range_y
                    img_x = torch.linspace(min_x,max_x,self.W).unsqueeze(0).repeat(self.H, 1).reshape(-1,1)
                    img_y = torch.linspace(min_y,max_y,self.H).unsqueeze(1).repeat(1,self.W).reshape(-1,1)
                    rays_xy = torch.cat((img_x,img_y), -1).unsqueeze(0).to(self.device)
                    ray_bundle = _xy_to_ray_bundle(camera, rays_xy, self.near, self.far, 64)
                    rays_o = ray_bundle.origins.squeeze(0)
                    rays_d = ray_bundle.directions.squeeze(0)
                    H = self.H 
                    W = self.W
                    rays_o = rays_o.split(128)
                    rays_d = rays_d.split(128)
                    out_rgb_fine = []
                    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                        render_out = self.renderer.render(rays_o_batch,
                                                        rays_d_batch,
                                                        self.near, self.far,
                                                        bone_transformation_inv, T_pose_21,None,
                                                        obj_r.T,obj_t,)
                        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                        del render_out
                    img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255)
                    img_name = test_name_groups[i][0].replace('.pickle','.jpeg')
                    for j in range(img_fine.shape[-1]):
                        img_file = os.path.join(save_render_path,img_name)
                        print(img_file)
                        cv2.imwrite(img_file, img_fine[..., j])
        
if __name__ == '__main__':
    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.case, args.render)
    runner.fitting()

# python get_res.py --conf ./fit_confs/get_res_1.conf --case get_res_1 --gpu 0
# python get_res.py --conf ./fit_confs/get_res_12.conf --case get_res_12 --gpu 0
# python get_res.py --conf ./fit_confs/get_res_123.conf --case get_res_123 --gpu 0
# python get_res.py --conf ./fit_confs/get_res_1234.conf --case get_res_1234 --gpu 0
# python get_res.py --conf ./fit_confs/get_render_type0.conf --case render_res_type0 --gpu 0 --render True
# python get_res.py --conf ./fit_confs/get_render_type1.conf --case render_res_type1 --gpu 0 --render True
# python get_res.py --conf ./fit_confs/get_render_type12.conf --case render_res_type12 --gpu 0 --render True