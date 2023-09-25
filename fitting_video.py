import os
import time
import logging
import argparse
import numpy as np
import pickle
import torch
import cv2
import torch.nn.functional as F
import shutil
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from utils.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, RenderingNetwork_OBJ, SDFNetwork_OBJ, Embedding
from utils.renderer_batch import NeuSRenderer_fitting

from utils.utils import  rot6d_to_matrix, _xy_to_ray_bundle
from utils.dataset import fit_video_dataset, get_rays_xy, RayImageSampler
from pytorch3d.renderer import PerspectiveCameras
from halo_util.utils import convert_joints
from halo_util.converter_fit_batch import PoseConverter, transform_to_canonical

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, gpu_id=1):
        self.conf_path = conf_path
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
        self.iter_step = 0
        self.batch_size = self.conf.get_int('train.batch_size')
        self.near = self.conf['train.near']
        self.far = self.conf['train.far']
        self.is_continue = is_continue
        self.mode = mode
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

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def get_bound(self, verts):
        cur_verts = verts.cpu().detach().numpy()
        x_min = cur_verts[:,0].min() - 0.04
        x_max = cur_verts[:,0].max() + 0.04
        y_min = cur_verts[:,1].min() - 0.04
        y_max = cur_verts[:,1].max() + 0.04
        z_min = cur_verts[:,2].min() - 0.04
        z_max = cur_verts[:,2].max() + 0.04
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
        self.sdf_network_hand = SDFNetwork(self.barf_encoding,hand_barf_len,self.data_type,**self.conf['model.sdf_hand_network'], use_batch=True).to(self.device)
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

        def pose_loss(target_pose, pred_pose):
            cur_err = torch.norm(target_pose-pred_pose, dim=-1)
            pose_error = cur_err.mean()
            return pose_error
        
        get_render_all = False
        sequence_list_file = './sequence_list_for_fitting.pickle'
        with open(sequence_list_file,'rb') as f:
            sequence_list = pickle.load(f)
        seq_num = len(sequence_list)
        for seq_id in range(seq_num):
            if seq_id != self.fit_id:
                continue
            load_model = False
            cur_seq = sequence_list[seq_id]
            obj_name = cur_seq['obj_name']
            frame_name = cur_seq['frame_name']
            self.fit_dataset = fit_video_dataset(
                data_root = self.data_root,
                fit_type = self.fit_type,
                obj_name = obj_name,
                frame_name = frame_name
                )
            sampler = RayImageSampler(self.fit_dataset, N_images=4,
                                N_iter=len(self.fit_dataset) - 3, device = self.device)
            self.fit_dataloader = torch.utils.data.DataLoader(
                self.fit_dataset, batch_sampler=sampler)
            self.get_res_dataloader = torch.utils.data.DataLoader(
                self.fit_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator(device='cuda'))
            self.file_backup()
            iter_num = 5
            data_num = len(self.fit_dataset)
            self.obj_rot_refine = torch.zeros(data_num, 3, 2)
            self.obj_rot_refine[:,0,0] = 1.0
            self.obj_rot_refine[:,1,1] = 1.0
            self.obj_rot_refine = self.obj_rot_refine.to(self.device)
            self.obj_rot_refine = torch.nn.Parameter(self.obj_rot_refine, requires_grad=True)
            self.obj_trans_refine = torch.zeros(data_num,3).to(self.device)
            self.obj_trans_refine = torch.nn.Parameter(self.obj_trans_refine, requires_grad=True)
            self.palm_rot_refine = torch.zeros(data_num, 3, 2)
            self.palm_rot_refine[:,0,0] = 1.0
            self.palm_rot_refine[:,1,1] = 1.0
            self.palm_rot_refine = self.palm_rot_refine.to(self.device)
            self.palm_rot_refine = torch.nn.Parameter(self.palm_rot_refine, requires_grad=True)
            self.palm_trans_refine = torch.zeros(data_num,3).to(self.device)
            self.palm_trans_refine = torch.nn.Parameter(self.palm_trans_refine, requires_grad=True)
            self.joint_refine_angle = torch.zeros(data_num,20).to(self.device)
            self.joint_refine_angle = torch.nn.Parameter(self.joint_refine_angle, requires_grad=True)
            self.palm_refine_angle = torch.zeros(data_num,7).to(self.device)
            self.palm_refine_angle = torch.nn.Parameter(self.palm_refine_angle, requires_grad=True)
            params_first_stage = [
            {"params": self.obj_rot_refine, "lr": 0.0001},
            {"params": self.obj_trans_refine, "lr": 0.0001},
            {"params": self.palm_rot_refine, "lr": 0.0001},
            {"params": self.palm_trans_refine, "lr": 0.0001},
            {"params": self.joint_refine_angle, "lr": 0.0001},
            {"params": self.palm_refine_angle, "lr": 0.0005},
            ]
            optimizer=torch.optim.Adam(params_first_stage)
            for iter_id in tqdm(range(iter_num)):
                for batch_idx, test_batch in enumerate(self.fit_dataloader):

                    image_group,mask_group,param_group,name_group,\
                    pred_joint_group, pred_objpose_group, proj_group, \
                    T_pose_21, bone_length,\
                    obj_verts,\
                    save_base_path,hand_model_path,obj_model_path,\
                    index = test_batch
                    if load_model == False:
                        self.init_model(hand_model_path[0], obj_model_path[0])
                        load_model = True
                    intial_param = param_group[0]
                    view_num = len(image_group[0])
                    Ro_gt = intial_param['obj_R'].to(self.device).float()  
                    To_gt = intial_param['obj_T'].to(self.device).float()  
                    joint3d_gt = intial_param['joint3d_21'].float().to(self.device) 
                    Ro_pred = pred_objpose_group[:,0,:3,:3].to(self.device).float()
                    To_pred = pred_objpose_group[:,0,:3,3].to(self.device).float()
                    joint3d_pred = pred_joint_group[:,0].to(self.device).float()
                    T_pose_21 = T_pose_21.to(self.device).float()
                    cur_bone_length = bone_length.to(self.device).float()
                    proj = proj_group.to(self.device).float()
                    obj_verts = obj_verts.to(self.device).float()
                    print(obj_name, frame_name)
                    for sub_iter_id in range(4):
                        for view_id in range(view_num):
                            R =  param_group[view_id]['cam_R'].float()
                            T = param_group[view_id]['cam_T'].float()
                            fx_ndc = param_group[view_id]['fx_ndc']
                            fy_ndc = param_group[view_id]['fy_ndc']
                            focal_length = torch.stack([fx_ndc, fy_ndc], 1).float()
                            px_ndc = param_group[view_id]['px_ndc']
                            py_ndc = param_group[view_id]['py_ndc']
                            principal_point = torch.stack([px_ndc, py_ndc], 1).float()
                            camera = PerspectiveCameras(R = R, T = T,focal_length = focal_length,\
                                                    principal_point = principal_point).to(self.device)
                            cur_obj_rot_refine = self.obj_rot_refine[index]
                            cur_obj_trans_refine = self.obj_trans_refine[index]
                            cur_palm_rot_refine = self.palm_rot_refine[index]
                            cur_palm_trans_refine = self.palm_trans_refine[index]
                            cur_joint_refine_angle = self.joint_refine_angle[index]
                            cur_palm_refine_angle = self.palm_refine_angle[index]
                            kps_local_cs = convert_joints(joint3d_pred, source='mano', target='biomech')
                            is_right_one = torch.ones(joint3d_pred.shape[0], device=kps_local_cs.device)
                            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                            joint_3d = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length,
                                        joint_refine_angle=cur_joint_refine_angle, palm_refine_angle=cur_palm_refine_angle*0.1)
                            glo_rot_right_inv = torch.inverse(glo_rot_right)  
                            joint_3d = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ joint_3d.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
                            hand_rots = rot6d_to_matrix(cur_palm_rot_refine)
                            R_palm = hand_rots 
                            T_palm = cur_palm_trans_refine 
                            joint_3d_root = joint_3d[:,:1,:].clone()
                            joint_3d = (R_palm.unsqueeze(1) @ (joint_3d - joint_3d_root).unsqueeze(-1))[...,0] + joint_3d_root + T_palm.unsqueeze(1)
                            kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech').to(self.device)
                            is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                            rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                            trans_mat_pc, _,_ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                            trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                            trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                            trans_mat_pc = trans_mat_pc_all
                            bone_transformation_inv = trans_mat_pc
                            obj_rots = rot6d_to_matrix(cur_obj_rot_refine)
                            obj_trans = cur_obj_trans_refine
                            obj_r = obj_rots @ Ro_pred
                            obj_t = To_pred + obj_trans
                            pred_obj_v_w = (obj_r.unsqueeze(1) @ obj_verts.unsqueeze(-1))[...,0] + obj_t.unsqueeze(1) 
                            compare_obj_v_w = (Ro_pred.unsqueeze(1) @ obj_verts.unsqueeze(-1))[...,0] + To_pred.unsqueeze(1)
                            obj_verts_loss = pose_loss(pred_obj_v_w, compare_obj_v_w)
                            gt_obj_v_w = (Ro_gt.unsqueeze(1) @ obj_verts.unsqueeze(-1))[...,0] + To_gt.unsqueeze(1)
                            obj_verts_err_to_gt = pose_loss(pred_obj_v_w, gt_obj_v_w)                          
                            cur_img = image_group[:,view_id]
                            cur_mask = mask_group[:,view_id]
                            rays_xy = []
                            true_rgb = []
                            true_mask = []
                            for cid in range(4):
                                mask_xy = np.where((cur_mask[cid,:,:,0] > 0))
                                cur_rays_xy, cur_true_rgb, cur_true_mask = get_rays_xy(cur_img[cid], cur_mask[cid], mask_xy, 40, threshold=1.0)
                                rays_xy.append(cur_rays_xy)
                                true_rgb.append(cur_true_rgb)
                                true_mask.append(cur_true_mask)
                            rays_xy = torch.stack(rays_xy, 0)
                            true_rgb = torch.stack(true_rgb, 0)
                            true_mask = torch.stack(true_mask, 0)
                            true_rgb = true_rgb.to(self.device)
                            true_mask = true_mask.to(self.device)
                            rays_xy = torch.FloatTensor(rays_xy).to(self.device)
                            rays_xy = rays_xy.float()
                            ray_bundle = _xy_to_ray_bundle(camera, rays_xy, self.near, self.far , 64)
                            rays_o = ray_bundle.origins
                            rays_d = ray_bundle.directions
                            render_out = self.renderer.render(rays_o,
                                                            rays_d,
                                                            self.near, self.far,
                                                            bone_transformation_inv, T_pose_21,None,
                                                            torch.inverse(obj_r), obj_t,)
                            color_fine = render_out['color_fine']
                            weight_sum = render_out['weight_sum']
                            color_error = (color_fine - true_rgb) * true_mask
                            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / true_mask.shape[0] / true_mask.shape[1]
                            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), true_mask)
                            render_loss = color_fine_loss + 0.5 * mask_loss
                            render_loss = 0.5 * render_loss
                            joint_loss = pose_loss(joint_3d, joint3d_pred)
                            joint_err_to_gt = pose_loss(joint_3d, joint3d_gt)
                            pose_refine_loss = 30 * joint_loss  + 20 * obj_verts_loss
                            sdf_hand = render_out['sdf_hand'][:,0]
                            sdf_obj = render_out['sdf_obj'][:,0]
                            sdf_abs_sum = torch.abs(sdf_hand) + torch.abs(sdf_obj)
                            contact_id = (sdf_abs_sum < 1e-2)
                            contact_sdf = sdf_abs_sum[contact_id]
                            contact_num = contact_id.float().sum() + 1e-9
                            contact_loss = torch.sum(contact_sdf) / contact_num
                            obj_inner_id = (sdf_obj<0)
                            hand_select_sdf = sdf_hand[obj_inner_id]
                            obj_select_sdf = sdf_obj[obj_inner_id]
                            penet_points_id = (hand_select_sdf<0)
                            penet_sdf = torch.abs(hand_select_sdf[penet_points_id]) + torch.abs(obj_select_sdf[penet_points_id])
                            penet_num = penet_points_id.float().sum() + 1e-9
                            penet_loss = torch.sum(penet_sdf) / penet_num
                            interaction_loss = 30 * contact_loss + 20*penet_loss 
                            joint_smooth = pose_loss(joint_3d[1:], joint_3d[:-1])
                            obj_smooth = pose_loss(pred_obj_v_w[1:], pred_obj_v_w[:-1])
                            smooth_loss = joint_smooth + obj_smooth
                            if iter_id + sub_iter_id + view_id > 0 and index[0] == 0:
                                joint_to_first_smooth = pose_loss(joint_3d[:1], joint3d_pred[:1])
                                obj_to_first_smooth = pose_loss(pred_obj_v_w[:1], compare_obj_v_w[:1])
                                smooth_loss = smooth_loss +  joint_to_first_smooth +  obj_to_first_smooth
                            elif iter_id + sub_iter_id + view_id > 0 and index[3] == data_num - 1:
                                joint_to_last_smooth = pose_loss(joint_3d[-1:], joint3d_pred[-1:])
                                obj_to_last_smooth = pose_loss(pred_obj_v_w[-1:], compare_obj_v_w[-1:])
                                smooth_loss = smooth_loss +  joint_to_last_smooth +  obj_to_last_smooth
                            smooth_loss *= 50
                            if self.fit_type == '1234':
                                stable_loss = self.renderer.get_stable_loss_cross(obj_verts, bone_transformation_inv,T_pose_21,obj_r,obj_t)
                                stable_loss = stable_loss * 100
                                loss = render_loss + interaction_loss + pose_refine_loss + smooth_loss + stable_loss
                                print('iter: %d, index: %d, view: %d  loss_all: %lf,color_loss: %lf,  mask_loss: %lf,joint_loss: %lf, obj_verts_loss: %lf,  gt_joint_loss: %lf, gt_obj_verts_loss: %lf, smooth_loss: %lf,  contact_loss: %lf,  penet_loss: %lf,  stable_loss: %lf'\
                                %( iter_id, index[0], view_id,loss,\
                                    color_fine_loss, mask_loss,\
                                    joint_loss, obj_verts_loss,joint_err_to_gt,obj_verts_err_to_gt,\
                                    smooth_loss,\
                                    contact_loss, penet_loss, stable_loss))
                            else:
                                loss = render_loss + interaction_loss + pose_refine_loss + smooth_loss
                                print('iter: %d, index: %d, view: %d  loss_all: %lf,color_loss: %lf,  mask_loss: %lf,joint_loss: %lf, obj_verts_loss: %lf,  gt_joint_loss: %lf, gt_obj_verts_loss: %lf, smooth_loss: %lf,  contact_loss: %lf,  penet_loss: %lf'\
                                    %( iter_id, index[0], view_id,loss,\
                                        color_fine_loss, mask_loss,\
                                        joint_loss, obj_verts_loss,joint_err_to_gt,obj_verts_err_to_gt,\
                                        smooth_loss,\
                                        contact_loss, penet_loss))
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            del render_out
                    
                if get_render_all and iter_id + 1 == iter_num:
                    get_render = True
                else:
                    get_render = False

                bt_inv_list = []
                ro_list = []
                to_list = []
                param_list = []
                name_list = []
                for batch_idx, test_batch in enumerate(self.get_res_dataloader):

                    image_group,mask_group,param_group,name_group,\
                    pred_joint_group, pred_objpose_group, proj_group, \
                    T_pose_21, bone_length,\
                    obj_verts,\
                    save_base_path,hand_model_path,obj_model_path,\
                    index = test_batch
                    print(index)
                    save_base_path = save_base_path[0]
                    pose_path = os.path.join(save_base_path, 'pose_' + str(iter_id))
                    os.makedirs(pose_path, exist_ok=True)
                    render_path = os.path.join(save_base_path, 'render_' + str(iter_id))
                    os.makedirs(render_path, exist_ok=True)
                    if load_model == False:
                        self.init_model(hand_model_path[0], obj_model_path[0])
                        load_model = True
                    cid_name = name_group[0][0].split('_')[0]
                    pose_file = os.path.join(pose_path,'{}.pickle'.format(cid_name))
                    Ro_pred = pred_objpose_group[:,0,:3,:3].to(self.device).float()
                    To_pred = pred_objpose_group[:,0,:3,3].to(self.device).float()
                    joint3d_pred = pred_joint_group[:,0].to(self.device).float()
                    T_pose_21 = T_pose_21.to(self.device).float()
                    cur_bone_length = bone_length.to(self.device).float()
                    obj_verts = obj_verts.to(self.device).float()
                    cur_obj_rot_refine = self.obj_rot_refine[index]
                    cur_obj_trans_refine = self.obj_trans_refine[index]
                    cur_palm_rot_refine = self.palm_rot_refine[index]
                    cur_palm_trans_refine = self.palm_trans_refine[index]
                    cur_joint_refine_angle = self.joint_refine_angle[index]
                    cur_palm_refine_angle = self.palm_refine_angle[index]
                    kps_local_cs = convert_joints(joint3d_pred, source='mano', target='biomech')
                    is_right_one = torch.ones(joint3d_pred.shape[0], device=kps_local_cs.device)
                    palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                    joint_3d = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length,
                                joint_refine_angle=cur_joint_refine_angle, palm_refine_angle=cur_palm_refine_angle*0.1) 
                    glo_rot_right_inv = torch.inverse(glo_rot_right)   
                    joint_3d = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ joint_3d.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
                    hand_rots = rot6d_to_matrix(cur_palm_rot_refine)
                    R_palm = hand_rots 
                    T_palm = cur_palm_trans_refine 
                    joint_3d_root = joint_3d[:,:1,:].clone()
                    joint_3d = (R_palm.unsqueeze(1) @ (joint_3d - joint_3d_root).unsqueeze(-1))[...,0] + joint_3d_root + T_palm.unsqueeze(1)
                    kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech').to(self.device)
                    is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                    palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                    rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                    trans_mat_pc, _,_ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                    trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                    trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                    trans_mat_pc = trans_mat_pc_all
                    bone_transformation_inv = trans_mat_pc
                    obj_rots = rot6d_to_matrix(cur_obj_rot_refine)
                    obj_trans = cur_obj_trans_refine
                    obj_r = obj_rots @ Ro_pred
                    obj_t = To_pred + obj_trans
                    bt_inv_list.append(bone_transformation_inv)
                    ro_list.append(obj_r)
                    to_list.append(obj_t)
                    param = {}
                    param['pred_Ro'] = obj_r.detach().cpu().numpy().copy()[0]
                    param['pred_To'] = obj_t.detach().cpu().numpy().copy()[0]
                    param['pred_joint3d'] = joint_3d.detach().cpu().numpy().copy()[0]
                    intial_param = param_group[0]
                    param_list.append(intial_param)
                    param['gt_Ro'] = intial_param['obj_R'].cpu().numpy()[0]
                    param['gt_To'] = intial_param['obj_T'].cpu().numpy()[0]
                    param['gt_joint3d'] = intial_param['joint3d_21'].cpu().numpy()[0]
                    f = open(pose_file,'wb')
                    pickle.dump(param,f)
                    f.close()
                    if get_render:
                        for view_id in range(1):
                            R =  param_group[view_id]['cam_R'].float()[0:1]
                            T = param_group[view_id]['cam_T'].float()[0:1]
                            fx_ndc = param_group[view_id]['fx_ndc'][0:1]
                            fy_ndc = param_group[view_id]['fy_ndc'][0:1]
                            focal_length = torch.stack([fx_ndc, fy_ndc], 1).float()
                            px_ndc = param_group[view_id]['px_ndc'][0:1]
                            py_ndc = param_group[view_id]['py_ndc'][0:1]
                            principal_point = torch.stack([px_ndc, py_ndc], 1).float()
                            camera = PerspectiveCameras(R = R, T = T,focal_length = focal_length,\
                                                    principal_point = principal_point).to(self.device)
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
                            rays_o = rays_o.split(220)
                            rays_d = rays_d.split(220)
                            out_rgb_fine = []
                            our_mask_fine = []
                            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                                render_out = self.renderer.render(rays_o_batch.unsqueeze(0),
                                                                rays_d_batch.unsqueeze(0),
                                                                self.near, self.far,
                                                                bone_transformation_inv[:1], T_pose_21[:1],None,
                                                                torch.inverse(obj_r[:1]),obj_t[:1],)
                                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy()[0])
                                our_mask_fine.append(render_out['weight_sum'].detach().cpu().numpy()[0])
                                del render_out
                            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255)
                            mask_fine = (np.concatenate(our_mask_fine, axis=0).reshape([H, W, -1]) * 255).clip(0, 255)
                            img_name = name_group[view_id][0].replace('.pickle','.jpeg')
                            print(img_name)
                            for j in range(img_fine.shape[-1]):
                                cv2.imwrite(os.path.join(render_path,img_name), img_fine[..., j])

if __name__ == '__main__':
    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='fitting')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.gpu)
    runner.fitting()

# python fitting_video.py --conf ./fit_confs/fit_123_8views_0.conf --case 123_8view_id0 --gpu 0
# python fitting_video.py --conf ./fit_confs/fit_1234_8views_0.conf --case 1234_8view_id0 --gpu 0
