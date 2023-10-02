import os
import time
import logging
import argparse
import numpy as np
import cv2
import trimesh
import math
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from utils.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, RenderingNetwork_OBJ, SDFNetwork_OBJ, Embedding, VGGLoss
from utils.renderer import NeuSRenderer
from utils.utils import rot6d_to_matrix
from utils.dataset import TrainDataLoad, TestDataLoad
from pytorch3d.renderer import PerspectiveCameras
from utils.utils import _xy_to_ray_bundle
from halo_util.utils import convert_joints
from halo_util.converter_fit_batch import PoseConverter, transform_to_canonical

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.device = torch.device('cuda')
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.model_type = self.conf.get_string('general.model_type')
        self.data_type = self.conf.get_string('general.data_type')
        self.H, self.W = self.conf.get_list('dataset.image_size')
        self.near = self.conf['train.near']
        self.far = self.conf['train.far']
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.refine_pose = self.conf.get_bool('train.refine_pose')
        self.H_prime = int(np.sqrt(self.batch_size))
        train_dataset = TrainDataLoad(
            data_root = self.conf.get_string('dataset.traindata_dir'),
            n_rays_per_image = self.conf.get_int('train.batch_size'),
            data_type = self.data_type,
            model_type = self.model_type,
            )
        test_dataset = TestDataLoad(
            data_root = self.conf.get_string('dataset.testdata_dir'),
            data_type = self.data_type,
            model_type=self.model_type,
            )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator(device='cuda')
            )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator(device='cuda')
            )
        self.iter_step = 0
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.vgg_weight = self.conf.get_float('train.vgg_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.barf_encoding = Embedding().to(self.device)
        self.pose_converter = PoseConverter(dev=self.device)
        params_to_train = []    
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        data_len = len(self.train_dataloader)
        if self.model_type == 'obj':
            self.sdf_network = SDFNetwork_OBJ(self.barf_encoding,data_len,self.data_type,**self.conf['model.sdf_network']).to(self.device)
            self.color_network = RenderingNetwork_OBJ(self.barf_encoding,self.data_type,**self.conf['model.rendering_network']).to(self.device)
            self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.model_type,
                                     **self.conf['model.neus_renderer'])
        else:
            self.sdf_network = SDFNetwork(self.barf_encoding,data_len,self.data_type,**self.conf['model.sdf_network']).to(self.device)
            self.color_network = RenderingNetwork(self.barf_encoding,self.data_type,**self.conf['model.rendering_network']).to(self.device)
            self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.model_type,
                                     **self.conf['model.neus_renderer'])

        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.vggloss = VGGLoss(self.device)
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        train_data_num = len(self.train_dataloader)
        end_epoch = math.ceil(self.end_iter / train_data_num)
        start_epoch = math.floor(self.iter_step / train_data_num)
        res_step = end_epoch - start_epoch
        self.update_learning_rate()
        vgg_start = self.end_iter * 0.3
        for iter_i in tqdm(range(res_step)):
            for iteration, batch in enumerate(self.train_dataloader):

                image,mask,R,T,focal_length,principal_point,Ro,To,verts,\
                random_sample, patch_sample, T_pose_21, cur_bone_length,index = batch
                if self.iter_step > vgg_start :
                    rays_xy, true_rgb, true_mask = patch_sample
                else:
                    rays_xy, true_rgb, true_mask = random_sample
                rays_xy = rays_xy.to(self.device)
                true_rgb = true_rgb.to(self.device)[0]
                true_mask = true_mask.to(self.device)[0]
                Ro = Ro.to(self.device)[0]
                To = To.to(self.device)[0]
                T_pose_21 = T_pose_21.to(self.device)[0]
                cur_bone_length = cur_bone_length.to(self.device)
                joint_3d = verts.to(self.device) 
                if self.model_type == 'obj':
                    bone_transformation_inv = torch.zeros((21,4,4)).to(self.device)
                    T_pose_21 = torch.zeros((21,3)).to(self.device)
                    if self.data_type == 'real' and self.refine_pose:
                        cur_refine_param = self.renderer.sdf_network.se3_refine[index]
                        rot_refine = cur_refine_param[0,:6]
                        trans_refine = cur_refine_param[0,6:9] * 0.1
                        obj_rots = rot6d_to_matrix(rot_refine)[0]
                        Ro = torch.matmul(obj_rots, Ro)
                        To = To + trans_refine
                else:
                    if self.data_type == 'real' and self.refine_pose:
                        cur_refine_param = self.renderer.sdf_network.se3_refine[index]  
                        palm_rot_refine = cur_refine_param[:,:6]
                        palm_trans_refine = cur_refine_param[:,6:9] * 0.1
                        joint_refine_angle = cur_refine_param[:,9:29]
                        palm_refine_angle = cur_refine_param[:,29:36] * 0.1
                        kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech')
                        is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                        joint_3d = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length,
                                                joint_refine_angle=joint_refine_angle, palm_refine_angle=palm_refine_angle)
                        glo_rot_right_inv = torch.inverse(glo_rot_right)
                        joint_3d = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ joint_3d.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
                        hand_rots = rot6d_to_matrix(palm_rot_refine)
                        R_palm = hand_rots
                        T_palm = palm_trans_refine
                        joint_3d_root = joint_3d[:,:1,:].clone()
                        refine_3d_joint = (R_palm.unsqueeze(1) @ (joint_3d - joint_3d_root).unsqueeze(-1))[...,0] + joint_3d_root + T_palm.unsqueeze(1)
                        kps_local_cs = convert_joints(refine_3d_joint, source='mano', target='biomech').cuda()
                        is_right_one = torch.ones(refine_3d_joint.shape[0], device=kps_local_cs.device)
                        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                        rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                        trans_mat_pc, _, _ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                        trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                        trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                        trans_mat_pc = trans_mat_pc_all
                        bone_transformation_inv = trans_mat_pc[0]
                    else:
                        kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech').cuda()
                        is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                        rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                        trans_mat_pc, _, _ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                        trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                        trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                        trans_mat_pc = trans_mat_pc_all
                        bone_transformation_inv = trans_mat_pc[0]

                camera = PerspectiveCameras(R = R, T = T,\
                    focal_length = focal_length, principal_point = principal_point).to(self.device)
                ray_bundle = _xy_to_ray_bundle(camera, rays_xy, self.near, self.far, 64)
                rays_o = ray_bundle.origins.squeeze(0)
                rays_d = ray_bundle.directions.squeeze(0)
                true_mask = (true_mask > 0.5).float()
                mask_sum = true_mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, self.near, self.far,
                                                  bone_transformation_inv,T_pose_21,
                                                  verts[0],
                                                  Ro.T,To,
                                                  index,
                                                  )
                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                color_error = (color_fine - true_rgb) * true_mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * true_mask).sum() / (mask_sum * 3.0)).sqrt())
                eikonal_loss = gradient_error
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), true_mask)
                loss = color_fine_loss +\
                        mask_loss * self.mask_weight
                loss = loss + eikonal_loss * self.igr_weight
                if self.iter_step > vgg_start and self.vgg_weight > 0. :
                    pred_img = color_fine.reshape((self.H_prime, self.H_prime,3)).permute(2,1,0).unsqueeze(0)
                    gt_img = true_rgb.reshape((self.H_prime, self.H_prime,3)).permute(2,1,0).unsqueeze(0)
                    if self.iter_step - vgg_start <= 10000.:
                        cur_iter_rate = (self.iter_step - vgg_start) / 10000.
                    else:
                        cur_iter_rate = 1.0
                    vgg_loss = self.vggloss(pred_img, gt_img)  # B,C,H,W
                    loss += cur_iter_rate * self.vgg_weight * vgg_loss
                else:
                    vgg_loss = 0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter_step += 1
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * true_mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * true_mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={} color_fine_loss={} eikonal_loss={} mask_loss={}, vgg_loss={}'.format(\
                            self.iter_step, loss, self.optimizer.param_groups[0]['lr'],\
                            color_fine_loss, eikonal_loss * self.igr_weight, \
                            mask_loss * self.mask_weight, vgg_loss))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                if self.iter_step % self.val_freq == 0:
                    self.validate_image()
                self.update_learning_rate()

                
    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

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

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'],strict=False)
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'],strict=False)
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'barf_encoding': self.barf_encoding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def test(self):
        
        os.makedirs(os.path.join(self.base_exp_dir, 'test_render'), exist_ok=True)
        for batch_idx, test_batch in enumerate(self.test_dataloader):
            test_R, test_T, focal_length,principal_point,Ro,To, T_pose_21,cur_bone_length,index, param_file_name, verts = test_batch
            Ro = Ro.to(self.device)[0]
            To = To.to(self.device)[0]
            cur_bone_length = cur_bone_length.to(self.device)
            joint_3d = verts.to(self.device) #[1,21,3]
            img_name = param_file_name[0].replace('.pickle','.jpeg')
            if self.model_type == 'hand':
                kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech').to(self.device)
                is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                trans_mat_pc, _,_ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                trans_mat_pc = trans_mat_pc_all
                bone_transformation_inv = trans_mat_pc[0]
                ones = torch.ones((21,1)).to(self.device)
                j_21_homo = torch.cat((joint_3d[0], ones), -1).unsqueeze(-1).to(self.device)
                j_21_ori = torch.matmul(trans_mat_pc[0], j_21_homo)[:,:3,0]
                T_pose_21 = j_21_ori
            else:
                bone_transformation_inv = torch.zeros((21,4,4)).to(self.device)
                T_pose_21 = torch.zeros((21,3)).to(self.device)

            test_camera = PerspectiveCameras(R = test_R, T = test_T,\
                        focal_length = focal_length, principal_point = principal_point).to(self.device)
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
            ray_bundle = _xy_to_ray_bundle(test_camera, rays_xy, self.near, self.far, 64)
            rays_o = ray_bundle.origins.squeeze(0)
            rays_d = ray_bundle.directions.squeeze(0)
            H = self.H 
            W = self.W
            rays_o = rays_o.split(self.batch_size)
            rays_d = rays_d.split(self.batch_size)
            out_rgb_fine = []
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  self.near,
                                                  self.far,
                                                  bone_transformation_inv, T_pose_21,
                                                  joint_3d,
                                                  Ro.T,To,
                                                  index)
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                del render_out

            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
            for i in range(img_fine.shape[-1]):
                cv2.imwrite(os.path.join(self.base_exp_dir,
                                        'test_render',
                                        img_name),
                                        img_fine[..., i])

            print(img_name)


    def validate_image(self, idx=-1, resolution_level=-1):

        for batch_idx, batch in enumerate(self.test_dataloader):
            test_batch = batch
            break
        test_R, test_T, focal_length,principal_point,Ro,To, T_pose_21,cur_bone_length,index, param_file_name, verts  = test_batch
        Ro = Ro.to(self.device)[0]
        To = To.to(self.device)[0]
        T_pose_21 = T_pose_21.to(self.device)[0]
        cur_bone_length = cur_bone_length.to(self.device)
        joint_3d = verts.to(self.device) #[1,21,3]
        if self.model_type == 'hand':
            kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech')
            is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
            refine_3d_joint = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length)
            glo_rot_right_inv = torch.inverse(glo_rot_right)
            refine_3d_joint = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ refine_3d_joint.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
            kps_local_cs = convert_joints(refine_3d_joint, source='mano', target='biomech').cuda()
            is_right_one = torch.ones(refine_3d_joint.shape[0], device=kps_local_cs.device)
            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
            rot_then_swap_mat = glo_rot_right.unsqueeze(1)
            trans_mat_pc, _, _ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
            trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
            trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
            trans_mat_pc = trans_mat_pc_all
            bone_transformation_inv = trans_mat_pc[0]
        else:
            bone_transformation_inv = torch.zeros((21,4,4)).to(self.device)
            T_pose_21 = torch.zeros((21,3)).to(self.device)
        test_camera = PerspectiveCameras(R = test_R, T = test_T,\
                        focal_length = focal_length, principal_point = principal_point).to(self.device)
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
        ray_bundle = _xy_to_ray_bundle(test_camera, rays_xy, self.near, self.far, 64)
        rays_o = ray_bundle.origins.squeeze(0)
        rays_d = ray_bundle.directions.squeeze(0)
        H = self.H 
        W = self.W
        rays_o = rays_o.split(self.batch_size)
        rays_d = rays_d.split(self.batch_size)
        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              self.near,
                                              self.far,
                                              bone_transformation_inv, T_pose_21,
                                              None,
                                              Ro.T,To,
                                              index)
            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv2.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                                        img_fine[..., i])


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):

        for batch_idx, batch in enumerate(self.test_dataloader):
            test_batch = batch
            test_R, test_T, focal_length,principal_point,Ro,To, T_pose_21,cur_bone_length,index, param_file_name, verts  = test_batch
            Ro = Ro.to(self.device)[0]
            To = To.to(self.device)[0]
            T_pose_21 = T_pose_21.to(self.device)[0]
            joint_3d = verts.to(self.device)
            cur_bone_length = cur_bone_length.to(self.device)
            mesh_name = param_file_name[0].replace('.pickle','.ply')
            if self.model_type == 'hand':
                kps_local_cs = convert_joints(joint_3d, source='mano', target='biomech')
                is_right_one = torch.ones(joint_3d.shape[0], device=kps_local_cs.device)
                palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                refine_3d_joint = self.pose_converter.get_refine_3d_joint(palm_align_kps_local_cs, is_right_one, cur_bone_length)
                glo_rot_right_inv = torch.inverse(glo_rot_right)
                refine_3d_joint = (glo_rot_right_inv[:,:3,:3].unsqueeze(1) @ refine_3d_joint.unsqueeze(-1))[...,0] + glo_rot_right_inv[:,:3,3].unsqueeze(1)
                kps_local_cs = convert_joints(refine_3d_joint, source='mano', target='biomech').cuda()
                is_right_one = torch.ones(refine_3d_joint.shape[0], device=kps_local_cs.device)
                palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
                rot_then_swap_mat = glo_rot_right.unsqueeze(1)
                trans_mat_pc, _, _ = self.pose_converter(palm_align_kps_local_cs, is_right_one)
                trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='mano')
                trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
                trans_mat_pc = trans_mat_pc_all
                bone_transformation_inv = trans_mat_pc[0]
            else:
                bone_transformation_inv = torch.zeros((21,4,4)).to(self.device)
                T_pose_21 = torch.zeros((21,3)).to(self.device)

            def get_bound(verts):
                cur_verts = verts.cpu().detach().numpy()
                if self.model_type == 'obj' and self.data_type == 'syn':
                    x_min = cur_verts[:,0].min() - 0.15
                    x_max = cur_verts[:,0].max() + 0.15
                    y_min = cur_verts[:,1].min() - 0.15
                    y_max = cur_verts[:,1].max() + 0.15
                    z_min = cur_verts[:,2].min() - 0.15
                    z_max = cur_verts[:,2].max() + 0.15
                else:
                    if self.model_type == 'hand':
                        x_min = cur_verts[:,0].min() - 0.15
                        x_max = cur_verts[:,0].max() + 0.15
                        y_min = cur_verts[:,1].min() - 0.15
                        y_max = cur_verts[:,1].max() + 0.15
                        z_min = cur_verts[:,2].min() - 0.15
                        z_max = cur_verts[:,2].max() + 0.15
                    else:
                        r = 0.2
                        x_min = 0 - r
                        x_max = 0 + r
                        y_min = 0 - r  
                        y_max = 0 + r
                        z_min = 0 - r 
                        z_max = 0 + r

                object_bbox_min = np.array([x_min, y_min, z_min])
                object_bbox_max = np.array([x_max, y_max, z_max])
                bound_min = torch.tensor(object_bbox_min, dtype=torch.float32)
                bound_max = torch.tensor(object_bbox_max, dtype=torch.float32)
                return bound_min, bound_max

            bound_min, bound_max = get_bound(joint_3d[0])

            vertices, triangles =\
                self.renderer.extract_geometry(bound_min, bound_max, resolution, 
                                                       bone_transformation_inv,T_pose_21,
                                                       Ro.T,To, threshold=0)
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            vertices = vertices * 1000
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(batch_idx)))

        logging.info('End')


if __name__ == '__main__':
    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'mesh':
        runner.validate_mesh(world_space=True, resolution=256, threshold=args.mcube_threshold)
    elif args.mode == 'test':
        runner.test()

# python exp_runner.py --mode train --conf ./confs/wmask_realobj_bean.conf --case bean --gpu 0
# python exp_runner.py --mode test --conf ./confs/wmask_realobj_bean.conf --case bean --gpu 0 --is_continue
# python exp_runner.py --mode mesh --conf ./confs/wmask_realobj_bean.conf --case bean --gpu 0 --is_continue
# python exp_runner.py --mode train --conf ./confs/wmask_realhand_hand1.conf --case hand1 --gpu 0
# python exp_runner.py --mode test --conf ./confs/wmask_realhand_hand1.conf --case hand1 --gpu 0 --is_continue
# python exp_runner.py --mode mesh --conf ./confs/wmask_realhand_hand1.conf --case hand1 --gpu 0 --is_continue
