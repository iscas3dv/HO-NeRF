import os
import time
import logging
import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import shutil
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from utils.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, RenderingNetwork_OBJ, SDFNetwork_OBJ, Embedding, VGGLoss
from utils.renderer import NeuSRenderer_fitting
from utils.utils import  rot6d_to_matrix, _xy_to_ray_bundle
from utils.dataset import fit_single_dataset, get_rays_xy
from pytorch3d.renderer import PerspectiveCameras
from halo_util.utils import convert_joints
from halo_util.converter_fit_batch import PoseConverter, transform_to_canonical

class Runner:
    def __init__(self, conf_path, case='CASE_NAME', gpu_id=1):
        self.conf_path = conf_path
        self.case = case
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
        self.view_num = self.conf.get_string('dataset.view_num')
        self.H, self.W = self.conf.get_list('dataset.image_size')
        fit_dataset = fit_single_dataset(
            data_root = self.data_root,
            view_num = self.view_num,
            fit_type = self.fit_type,
            )
        self.fit_dataloader = torch.utils.data.DataLoader(
            fit_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator(device='cuda')
            )
        self.iter_step = 0
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

        def pose_loss(target_pose, pred_pose):
            cur_err = torch.norm(target_pose-pred_pose, dim=-1)
            pose_error = cur_err.sum() / cur_err.shape[0]
            return pose_error

        if self.fit_type == '1':
            iter_num = 30
            if self.view_num == '3':
                iter_num = 40
        elif self.fit_type == '12':
            iter_num = 25
            if self.view_num == '3':
                iter_num = 35
        iter_num = int(iter_num)
        save_flag = 0
        for batch_idx, test_batch in enumerate(self.fit_dataloader):

            image_group,mask_group,param_group,test_param_group,name_group,test_name_groups,\
            pred_joint_group, pred_objpose_group, proj_group, \
            T_pose_21, bone_length,\
            obj_verts,\
            hand_model_path, obj_model_path,save_base_path,\
            index = test_batch
            save_base_path = save_base_path[0]
            self.init_model(hand_model_path[0], obj_model_path[0])
            if save_flag == 0:
                save_flag = 1
                self.file_backup(save_base_path[0])
            to_config_path = os.path.join(save_base_path, 'config')
            os.makedirs(to_config_path, exist_ok=True)
            to_config_file = os.path.join(to_config_path, 'config.conf')
            if not os.path.exists(to_config_file):
                shutil.copy(self.conf_path, to_config_file)
            pose_path = os.path.join(save_base_path, 'pose_' + self.fit_type)
            os.makedirs(pose_path, exist_ok=True)
            intial_param = param_group[0]
            cid_name = name_group[0][0].split('_')[0]
            pose_file = os.path.join(pose_path,'{}.pickle'.format(cid_name))
            if os.path.exists(pose_file):
                continue
            param = {}
            view_num = len(image_group[0])
            Ro_gt = intial_param['obj_R'].to(self.device)[0].float()  
            To_gt = intial_param['obj_T'].to(self.device)[0].float()  
            joint3d_gt = intial_param['joint3d_21'].float().to(self.device)[0]  
            param['gt_joint3d'] = joint3d_gt.detach().cpu().numpy().copy()
            param['gt_Ro'] = Ro_gt.detach().cpu().numpy().copy()
            param['gt_To'] = To_gt.detach().cpu().numpy().copy()
            Ro_pred = pred_objpose_group[0,0,:3,:3].to(self.device).float()
            To_pred = pred_objpose_group[0,0,:3,3].to(self.device).float()
            joint3d_pred = pred_joint_group[0,0].to(self.device).float()
            T_pose_21 = T_pose_21.to(self.device)[0].float()
            cur_bone_length = bone_length.to(self.device).float()
            obj_verts = obj_verts.to(self.device)[0].float()
            proj = proj_group[0].to(self.device).float()
            ori_3d_pose = joint3d_pred.unsqueeze(0)
            ori_obj_r = Ro_pred
            ori_obj_t = To_pred
            obj_rot_refine = torch.eye(3).to(self.device)
            obj_rot_refine = obj_rot_refine[:,:2]
            obj_rot_refine = torch.nn.Parameter(obj_rot_refine.float(), requires_grad=True)
            obj_trans_refine = torch.zeros(3).to(self.device)
            obj_trans_refine = torch.nn.Parameter(obj_trans_refine.float(), requires_grad=True)
            palm_rot_refine = torch.eye(3).to(self.device)
            palm_rot_refine = palm_rot_refine[:,:2]
            palm_rot_refine = torch.nn.Parameter(palm_rot_refine.unsqueeze(0).float(), requires_grad=True)
            palm_trans_refine = torch.zeros(3).to(self.device)
            palm_trans_refine = torch.nn.Parameter(palm_trans_refine.unsqueeze(0).float(), requires_grad=True)
            joint_refine_angle = torch.zeros(20).to(self.device)
            joint_refine_angle = torch.nn.Parameter(joint_refine_angle.unsqueeze(0).float(), requires_grad=True)
            palm_refine_angle = torch.zeros(7).to(self.device)
            palm_refine_angle = torch.nn.Parameter(palm_refine_angle.unsqueeze(0).float(), requires_grad=True)
            params_first_stage = [
                {"params": obj_rot_refine, "lr": 0.0005},
                {"params": obj_trans_refine, "lr": 0.0005},
                {"params": palm_rot_refine, "lr": 0.0005},
                {"params": palm_trans_refine, "lr": 0.0003},
                {"params": joint_refine_angle, "lr": 0.001},
                {"params": palm_refine_angle, "lr": 0.001},
                ]
            optimizer=torch.optim.Adam(params_first_stage)
            for iter_id in tqdm(range(iter_num)):
                for view_id in range(view_num):

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
                    trans_mat_pc, _, _ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                    trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                    trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                    trans_mat_pc = trans_mat_pc_all
                    bone_transformation_inv = trans_mat_pc[0]
                    obj_rots = rot6d_to_matrix(obj_rot_refine)[0]
                    obj_trans = obj_trans_refine
                    obj_r = obj_rots @ ori_obj_r
                    obj_t = ori_obj_t + obj_trans
                    pred_obj_v_w = (obj_r.unsqueeze(0) @ (obj_verts).unsqueeze(-1))[...,0] + obj_t
                    compare_obj_v_w = (Ro_pred.unsqueeze(0) @ (obj_verts).unsqueeze(-1))[...,0] + To_pred
                    obj_verts_loss = pose_loss(compare_obj_v_w, pred_obj_v_w)
                    gt_obj_v_w = (Ro_gt.unsqueeze(0) @ (obj_verts).unsqueeze(-1))[...,0] + To_gt
                    obj_verts_err_to_gt = pose_loss(gt_obj_v_w, pred_obj_v_w)
                    cur_img = image_group[0,view_id]
                    cur_mask = mask_group[0,view_id]
                    mask_xy = np.where((cur_mask[:,:,0] > 0))
                    rays_xy, true_rgb, true_mask = get_rays_xy(cur_img, cur_mask, mask_xy, self.batch_size, threshold=1.0)
                    true_rgb = true_rgb.to(self.device)
                    true_mask = true_mask.to(self.device)
                    rays_xy = torch.FloatTensor(rays_xy).to(self.device)
                    ray_bundle = _xy_to_ray_bundle(camera, rays_xy, self.near, self.far , self.batch_size)
                    rays_o = ray_bundle.origins.squeeze(0)
                    rays_d = ray_bundle.directions.squeeze(0)
                    render_out = self.renderer.render(rays_o,
                                                      rays_d,
                                                      self.near, self.far,
                                                      bone_transformation_inv, T_pose_21,None,
                                                      obj_r.T,obj_t,)
                    color_fine = render_out['color_fine']
                    weight_sum = render_out['weight_sum']
                    color_error = (color_fine - true_rgb) * true_mask
                    color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / true_mask.shape[0]
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), true_mask)
                    render_loss = color_fine_loss + 0.5 * mask_loss 
                    joint_loss = pose_loss(joint3d_pred, joint_3d[0])
                    joint_err_to_gt = pose_loss(joint3d_gt, joint_3d[0])
                    if self.fit_type == '1':
                        pose_refine_loss = 100* joint_loss + 5 * obj_verts_loss
                        loss = render_loss + pose_refine_loss
                        print('iter: %d,  loss_all: %lf,color_loss: %lf,  mask_loss: %lf,joint_loss: %lf, obj_verts_loss: %lf,  gt_joint_loss: %lf, gt_obj_verts_loss: %lf, '\
                         %( iter_id, loss,\
                            color_fine_loss, mask_loss,\
                            joint_loss, obj_verts_loss,joint_err_to_gt,obj_verts_err_to_gt))
                    elif self.fit_type == '12':
                        pose_refine_loss = 30 * joint_loss + 20 * obj_verts_loss
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
                        interaction_loss = 30 * contact_loss +20 * penet_loss
                        loss = render_loss + interaction_loss + pose_refine_loss
                        print('iter: %d,  loss_all: %lf,color_loss: %lf,  mask_loss: %lf,joint_loss: %lf, obj_verts_loss: %lf,  gt_joint_loss: %lf, gt_obj_verts_loss: %lf, contact_loss: %lf,  penet_loss: %lf'\
                         %( iter_id, loss,\
                            color_fine_loss, mask_loss,\
                            joint_loss, obj_verts_loss,joint_err_to_gt,obj_verts_err_to_gt,\
                            contact_loss, penet_loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            obj_rots = rot6d_to_matrix(obj_rot_refine)[0]
            obj_trans = obj_trans_refine
            obj_r = obj_rots @ ori_obj_r
            obj_t = ori_obj_t + obj_trans
            param['pred_Ro'] = obj_r.detach().cpu().numpy().copy()
            param['pred_To'] = obj_t.detach().cpu().numpy().copy()
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
            param['pred_joint3d'] = joint_3d[0].detach().cpu().numpy().copy()
            print(pose_file)
            f = open(pose_file,'wb')
            pickle.dump(param,f)
            f.close()
                       

if __name__ == '__main__':
    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.case, args.gpu)
    runner.fitting()

# python fitting_single.py --conf ./fit_confs/fit_1_8views.conf --case 1_8view --gpu 0
# python fitting_single.py --conf ./fit_confs/fit_12_8views.conf --case 12_8view --gpu 0
