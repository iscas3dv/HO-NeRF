import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) 
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) 
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class NeuSRenderer_fitting:
    def __init__(self,
                 sdf_network_hand,
                 deviation_network_hand,
                 color_network_hand,
                 sdf_network_obj,
                 deviation_network_obj,
                 color_network_obj,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):

        self.sdf_network_hand = sdf_network_hand
        self.deviation_network_hand = deviation_network_hand
        self.color_network_hand = color_network_hand
        self.sdf_network_obj = sdf_network_obj
        self.deviation_network_obj = deviation_network_obj
        self.color_network_obj = color_network_obj
        self.use_multiple_streams = True
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, pixel_samples,n_samples = z_vals.shape
        pts = rays_o[:,:, None, :]  + rays_d[:,:, None, :]  * z_vals[:, :,:,None]  
        sdf = sdf.reshape(self.batch_size, self.pixel_sample, n_samples) 
        prev_sdf, next_sdf = sdf[:, :,:-1], sdf[:,:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :,:-1], z_vals[:,:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5) 
        prev_cos_val = torch.cat([torch.zeros([batch_size,self.pixel_sample, 1]), cos_val[:, :,:-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)  
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([self.batch_size, self.pixel_sample,1]), 1. - alpha + 1e-7], -1), -1)[:, :,:-1]
        z_samples = torch.zeros((self.batch_size, self.pixel_sample, n_importance)).to(rays_o.device)
        for i in range(batch_size):
            z_samples[i] = sample_pdf(z_vals[i], weights[i], n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, bt_inv,T_pose_21,ctype,last=False):

        batch_size, pixel_sample, cur_n_samples = z_vals.shape
        _, _, n_importance = new_z_vals.shape
        pts = rays_o[:,:, None, :]  + rays_d[:,:, None, :] * new_z_vals[:, :,:,None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            if ctype == 'obj':
                new_sdf = self.sdf_network_obj.sdf(pts.reshape(-1, 3)).reshape(self.batch_size, self.pixel_sample,n_importance)
            else:
                new_sdf = self.sdf_network_hand.sdf(pts.reshape(self.batch_size,-1, 3), bt_inv, T_pose_21).reshape(self.batch_size, self.pixel_sample,n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1).reshape(self.batch_size * self.pixel_sample, cur_n_samples + n_importance)
            xx = torch.arange(self.pixel_sample)[None,:, None].expand(self.batch_size,self.pixel_sample, cur_n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(self.batch_size,self.pixel_sample, cur_n_samples + n_importance)

        return z_vals, sdf

    def get_alpha_sample_color( self,
                                rays_o, rays_d,
                                bt_inv,T_pose_21,
                                z_vals,
                                sample_dist,
                                ctype,
                                get_SDF=False):
        batch_size, pixel_sample, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, :,None, :] + rays_d[:, :,None, :] * mid_z_vals[..., None]  
        dirs = rays_d[:,:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        index = 0
        if ctype == 'obj':
            sdf_nn_output = self.sdf_network_obj(pts)
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]
            gradients = self.sdf_network_obj.gradient(pts).squeeze()
            sampled_color =self.color_network_obj(pts, dirs, feature_vector, gradients, index).reshape(batch_size,pixel_sample, n_samples, 3)
            inv_s = self.deviation_network_obj(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           
            inv_s = inv_s.expand(batch_size*pixel_sample*n_samples, 1)

        else:
            sdf_nn_output, xyz_feature, r,h = self.sdf_network_hand(pts.reshape(self.batch_size,-1, 3), bt_inv, T_pose_21)
            sdf = sdf_nn_output[:, :1] 
            feature_vector = sdf_nn_output[:, 1:] 

            gradients = self.sdf_network_hand.gradient(pts.reshape(self.batch_size,-1, 3), bt_inv, T_pose_21).squeeze().reshape(-1,3) 
            sampled_color = self.color_network_hand(dirs, xyz_feature,feature_vector,h, gradients,index)
            sampled_color = sampled_color.reshape(batch_size,pixel_sample, n_samples, 3)
            inv_s = self.deviation_network_hand(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)         
            inv_s = inv_s.expand(batch_size*pixel_sample*n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        cos_anneal_ratio = 1.0
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, pixel_sample, n_samples).clip(0.0, 1.0)

        gradients = gradients.reshape(batch_size, pixel_sample, n_samples, 3)
        gradient_error = (torch.linalg.norm(gradients, ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = gradient_error.mean()

        return alpha, sampled_color, sdf.reshape(-1,1), gradient_error, gradients.reshape(-1,3)

    def convert_obj_to_local(self, rays_o, rays_d, Ro, To):

        rays_o = rays_o - To.unsqueeze(1)
        rays_o = torch.matmul(Ro.unsqueeze(1), rays_o.unsqueeze(-1))[...,-1]
        rays_d = torch.matmul(Ro.unsqueeze(1), rays_d.unsqueeze(-1))[...,-1]

        return rays_o, rays_d

    def render( self, 
                rays_o, rays_d, 
                near, far, 
                bt_inv,T_pose_21,
                verts,
                Ro,To,
                get_SDF = False):

        self.batch_size, self.pixel_sample, _ = rays_o.shape
        rays_o_hand, rays_d_hand = rays_o.clone(), rays_d.clone()
        rays_o_obj, rays_d_obj = self.convert_obj_to_local(rays_o, rays_d, Ro, To)
        batch_size = len(rays_o)
        sample_dist = (far - near) / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]
        z_vals = z_vals.unsqueeze(1).repeat(self.batch_size,1,1)   
        perturb = self.perturb
        if perturb > 0:
            t_rand = (torch.rand([self.batch_size,self.pixel_sample, 1]) - 0.5)
            z_vals = z_vals + t_rand * sample_dist 
        z_vals_hand = z_vals  
        z_vals_obj = z_vals

        if self.n_importance > 0:
            with torch.no_grad():
                pts_hand = rays_o_hand[:,:, None, :] + rays_d_hand[:, :,None, :] * z_vals[:,:,:,None]
                pts_obj = rays_o_obj[:, :,None, :] + rays_d_obj[:,:, None, :] * z_vals[ :, :,:,None]
                sdf_hand = self.sdf_network_hand.sdf(pts_hand.reshape(batch_size,-1,3), bt_inv, T_pose_21).reshape(self.batch_size, self.pixel_sample,self.n_samples)
                sdf_obj = self.sdf_network_obj.sdf(pts_obj.reshape(-1, 3)).reshape(self.batch_size, self.pixel_sample,self.n_samples)
                
                for i in range(self.up_sample_steps):
                    new_z_vals_hand = self.up_sample(rays_o_hand,
                                                    rays_d_hand,
                                                    z_vals_hand,
                                                    sdf_hand,
                                                    self.n_importance // self.up_sample_steps,
                                                    64 * 2**i)
                    z_vals_hand, sdf_hand = self.cat_z_vals(rays_o_hand,
                                                      rays_d_hand,
                                                      z_vals_hand,
                                                      new_z_vals_hand,
                                                      sdf_hand,
                                                      bt_inv,T_pose_21,
                                                      'hand',
                                                      last=(i + 1 == self.up_sample_steps))
                    new_z_vals_obj = self.up_sample(rays_o_obj,
                                                    rays_d_obj,
                                                    z_vals_obj,
                                                    sdf_obj,
                                                    self.n_importance // self.up_sample_steps,
                                                    64 * 2**i)
                    z_vals_obj, sdf_obj = self.cat_z_vals(rays_o_obj,
                                                      rays_d_obj,
                                                      z_vals_obj,
                                                      new_z_vals_obj,
                                                      sdf_obj,
                                                      bt_inv,T_pose_21,
                                                      'obj',
                                                      last=(i + 1 == self.up_sample_steps))

                    z_vals = torch.cat([z_vals,new_z_vals_hand,new_z_vals_obj], dim=-1)
            n_samples = self.n_samples + 2 * self.n_importance
        z_vals, index = torch.sort(z_vals, dim=-1)
        alpha_hand, sample_color_hand, sdf_hand, gradient_error_hand,gradient_hand = self.get_alpha_sample_color(rays_o_hand, rays_d_hand,
                                                                        bt_inv,T_pose_21,
                                                                        z_vals,
                                                                        sample_dist,
                                                                        'hand')
        alpha_obj, sample_color_obj, sdf_obj, gradient_error_obj, gradient_obj = self.get_alpha_sample_color(rays_o_obj, rays_d_obj,
                                                                  bt_inv,T_pose_21,
                                                                  z_vals,
                                                                  sample_dist,
                                                                  'obj')

        final_alpha = (1. - alpha_hand + 1e-7) * (1. - alpha_obj + 1e-7)  

        weights_hand = alpha_hand * torch.cumprod(torch.cat([torch.ones([batch_size,self.pixel_sample, 1]), final_alpha], -1), -1)[..., :-1]
        weights_hand_sum = weights_hand.sum(dim=-1, keepdim=True)
        color_hand = (sample_color_hand * weights_hand[..., None]).sum(dim=-2)

        weights_obj = alpha_obj * torch.cumprod(torch.cat([torch.ones([batch_size, self.pixel_sample, 1]), final_alpha], -1), -1)[..., :-1]
        weights_obj_sum = weights_obj.sum(dim=-1, keepdim=True)
        color_obj = (sample_color_obj * weights_obj[..., None]).sum(dim=-2)

        color = color_hand + color_obj
        weights_sum = weights_hand_sum + weights_obj_sum
        weights_sum = weights_sum.sum(dim=-1, keepdim=True)

        return {
            'color_fine': color,
            'weight_sum': weights_sum,
            'sdf_hand': sdf_hand,
            'sdf_obj': sdf_obj,
            'gradient_error_hand': gradient_error_hand,
            'gradient_error_obj':gradient_error_obj,
            'gradient_hand': gradient_hand,
            'gradient_obj':gradient_obj
        }

    def extract_geometry(self, bound_min, bound_max, resolution, bt_inv, T_pose_21,Ro,To,get_type,threshold=0.0):

        print('threshold: {}'.format(threshold))
        N = 16
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).float()
                        if get_type == 'hand':
                            val = self.sdf_network_hand.sdf(pts.reshape(1,-1,3), bt_inv, T_pose_21).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        else:
                            obj_pts = pts - To
                            obj_pts = torch.matmul(Ro, obj_pts.unsqueeze(-1))[...,0]
                            val = self.sdf_network_obj.sdf(obj_pts.reshape(-1, 3)).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                        torch.cuda.empty_cache()

        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices, triangles = mcubes.marching_cubes(u, threshold)
        triangles = triangles[...,::-1]
        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

        return vertices, triangles


   

    def get_stable_loss_cross(self, pts, bt_inv, T_pose_21, Ro, To):
        pts = pts[:,::10,:]
        batch_size, p_num,_ = pts.shape
        pts_world = (Ro.unsqueeze(1) @ pts.unsqueeze(-1))[...,0] + To.unsqueeze(1)

        vert_id_all = range(p_num)

        hand_sdf = self.sdf_network_hand.sdf(pts_world, bt_inv, T_pose_21).reshape(batch_size, p_num, 1)

        hand_sdf_list = []
        in_id_list = []

        for batch_id in range(batch_size):
            cur_hand_sdf = hand_sdf[batch_id].reshape(-1)
            penet_id = (cur_hand_sdf<0)
            penet_num = penet_id.float().sum()

            if penet_num > 0:
                in_id_list.append(penet_id)
                hand_sdf_list.append(cur_hand_sdf)

        stable_loss = 0
        if len(in_id_list) > 1:

            hand_sdf_list = torch.stack(hand_sdf_list, 0)
            in_time = hand_sdf_list.shape[0]

            for cid in range(in_time):

                cur_in_id = in_id_list[cid].clone().cpu()
                cur_out_id = np.setdiff1d(vert_id_all, cur_in_id)

                in_points = pts[0,cur_in_id].clone().cpu()
                out_points = pts[0,cur_out_id].clone().cpu()

                in_points_num = in_points.shape[0]

                nn_index = spatial.cKDTree(out_points)
                nn_dists, near_out_id = nn_index.query(in_points, k=1)
                near_out_id = near_out_id.reshape(-1)
                near_out_id = np.unique(near_out_id)

                in_err = hand_sdf_list[:,cur_in_id].clip(0,1e7).sum() / ((in_time - 1) * in_points_num)


                hand_sdf_select = hand_sdf_list[:,cur_out_id]
                out_err = torch.abs(hand_sdf_select[:,near_out_id].clip(-1e7,0)).sum() / ((in_time - 1) * in_points_num)

                stable_loss = stable_loss + in_err + 0.05 * out_err

            stable_loss /= in_time


        return stable_loss
