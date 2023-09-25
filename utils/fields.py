import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# This implementation is borrowed from IDR: https://github.com/chenhsuanlin/bundle-adjusting-NeRF
class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=input.device) # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
    
def anerf_emb_point(pts, bt_inv, T_pose_21):
    tau = 200
    cutoff_dist = [[0.08,0.03,0.03,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02],]
    cutoff_dist = torch.Tensor(cutoff_dist).to(pts.device).unsqueeze(-1)
    pts2 = pts.unsqueeze(1).unsqueeze(-1) 
    bt_inv2 = bt_inv.unsqueeze(0)           
    cur_r2 = bt_inv2[...,:3,:3]  
    cur_t2 = bt_inv2[...,:3,3]  
    q = torch.matmul(cur_r2, pts2)[...,0] + cur_t2  
    q = q - T_pose_21[None,...]      
    v = torch.norm(q, dim=-1, p=2).unsqueeze(-1)   
    r = q / v    
    h = tau * (v - cutoff_dist)
    h = 1. - torch.sigmoid(h)   
    return v,r,h

def anerf_emb_point_batch(pts, bt_inv, T_pose_21):
    tau = 200
    cutoff_dist = [[0.08,0.03,0.03,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02,0.03,0.02,0.02,0.02],]
    cutoff_dist = torch.Tensor(cutoff_dist).to(pts.device).unsqueeze(-1)
    pts2 = pts.unsqueeze(2).unsqueeze(-1)   
    bt_inv2 = bt_inv.unsqueeze(1)           
    cur_r2 = bt_inv2[...,:3,:3]  
    cur_t2 = bt_inv2[...,:3,3]  
    q = torch.matmul(cur_r2, pts2)[...,0] + cur_t2  
    q = q - T_pose_21.unsqueeze(1)     
    v = torch.norm(q, dim=-1, p=2).unsqueeze(-1)    
    r = q / v     
    h = tau * (v - cutoff_dist)
    h = 1. - torch.sigmoid(h)   
    return v,r,h

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr

class SDFNetwork(nn.Module):
    def __init__(self,
                 barf_encoding,
                 traindata_num,
                 data_type,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 v_multires=10,
                 r_multires=4,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 use_batch=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.barf_encoding = barf_encoding
        self.embed_fn_fine = None
        self.v_multires = v_multires
        self.r_multires = r_multires
        self.data_type = data_type
        self.use_batch = use_batch
        v_in = 1
        if v_multires > 0:
            input_ch = self.v_multires * 2 * v_in + v_in
            input_ch_dir = self.r_multires * 2 * d_in + d_in
            dims[0] = (input_ch + input_ch_dir)*21

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):

            out_dim = dims[l + 1]
            if l in self.skip_in:
                lin = nn.Linear(dims[l] + dims[0], out_dim)
            else:
                lin = nn.Linear(dims[l] , out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif v_multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif v_multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        
        se3_refine = torch.zeros((traindata_num, 6+3+20+7))  # palm_rot, palm_trans, joint_angle, palm_angle
        se3_refine[:,0] = 1
        se3_refine[:,3] = 1
        self.se3_refine = nn.Parameter(se3_refine, requires_grad = True)

    def forward(self, x, bt_inv, T_pose_21):
        
        if self.use_batch:
            v,r,h = anerf_emb_point_batch(x, bt_inv, T_pose_21)
            v = v.reshape(-1,21,1)
            r = r.reshape(-1,21,3)
            h = h.reshape(-1,21,1)
        else:
            v,r,h = anerf_emb_point(x, bt_inv, T_pose_21)
            
        inputs_feature = self.barf_encoding(v, self.v_multires)
        sdf_emb_xyz = torch.cat([v,inputs_feature],dim=-1)
        dir_featrue = self.barf_encoding(r, self.r_multires)
        dir_emb = torch.cat([r, dir_featrue], dim=-1)
        sdf_emb_xyz = torch.cat((sdf_emb_xyz, dir_emb),-1) * h
        xyz_feature = sdf_emb_xyz.flatten(start_dim=-2, end_dim=-1)
        x = xyz_feature
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, xyz_feature], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] , x[:, 1:]], dim=-1), xyz_feature, r,h

    def sdf(self, x, bt_inv, T_pose_21):
        sdf_out, _, _, _ = self.forward(x, bt_inv, T_pose_21)
        return sdf_out[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, bt_inv, T_pose_21):
        x.requires_grad_(True)
        y = self.sdf(x,bt_inv, T_pose_21)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(self,
                 barf_encoding,
                 data_type,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 v_multires=10,
                 r_multires=4,
                 grad_multires=4,
                 squeeze_out=True,
                 use_gradients=False):
        super().__init__()
        self.barf_encoding = barf_encoding
        self.data_type = data_type
        self.squeeze_out = squeeze_out
        self.use_gradients = use_gradients
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        v_in = 1
        self.grad_multires = grad_multires
        self.v_multires = v_multires
        self.r_multires = r_multires
        input_ch = self.grad_multires * 2 * d_in + d_in
        input_ch_1 = self.v_multires * 2 * v_in + v_in
        input_ch_2 = self.r_multires * 2 * d_in + d_in
        dims[0] = (input_ch_1 + input_ch_2) * 21 + d_feature
        if self.use_gradients:
            dims[0] = dims[0] + input_ch

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, d, xyz_feature, feature_vector,h,gradients,index):

        view_feature = self.barf_encoding(gradients, self.grad_multires)
        color_emb_dir = torch.cat([gradients,view_feature],dim=-1)
        rendering_input = None
        rendering_input = torch.cat([xyz_feature,feature_vector], dim=-1)
        if self.use_gradients:
            rendering_input = torch.cat([rendering_input, color_emb_dir], dim = -1)

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

class SDFNetwork_OBJ(nn.Module):
    def __init__(self,
                 barf_encoding,
                 traindata_num,
                 data_type,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 v_multires=10,  #6
                 r_multires=4,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork_OBJ, self).__init__()
        
        self.v_multires = v_multires
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        input_ch = self.v_multires * 2 * d_in + d_in
        dims[0] = input_ch
        self.barf_encoding = barf_encoding
        self.data_type = data_type
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif self.v_multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.v_multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)
        se3_refine = torch.zeros((traindata_num, 6+3))  # rot, trans
        se3_refine[:,0] = 1
        se3_refine[:,3] = 1
        self.se3_refine = nn.Parameter(se3_refine, requires_grad = True)

    def forward(self, inputs):

        inputs_feature = self.barf_encoding(inputs, self.v_multires)
        inputs = torch.cat([inputs,inputs_feature],dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork_OBJ(nn.Module):
    def __init__(self,
                 barf_encoding,
                 data_type,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 v_multires=10,  
                 r_multires=4,
                 grad_multires=4,
                 squeeze_out=True,
                 use_gradients=False):
        super().__init__()

        self.barf_encoding = barf_encoding
        self.v_multires = v_multires
        self.r_multires = r_multires
        self.grad_multires = grad_multires
        self.squeeze_out = squeeze_out
        self.data_type = data_type
        self.use_gradients = use_gradients
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        input_ch_view = self.grad_multires * 2 * d_in + d_in
        input_ch = self.r_multires * 2 * d_in + d_in
        input_p = self.v_multires * 2 * d_in + d_in
        dims[0] = input_ch + input_p + d_feature + input_ch_view
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, points, view_dirs, feature_vectors,gradients,index):

        view_feature = self.barf_encoding(view_dirs, self.r_multires)
        view_dirs = torch.cat([view_dirs,view_feature],dim=-1)
        grad_emb = self.barf_encoding(gradients, self.grad_multires)
        grad_emb = torch.cat([gradients,grad_emb],dim=-1)
        point_emb = self.barf_encoding(points, self.v_multires)
        point_emb = torch.cat([points,point_emb],dim=-1)
        rendering_input = None
        rendering_input = torch.cat([point_emb, view_dirs,feature_vectors,grad_emb], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x

class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        feature_layers = (2,7,12,21,30)
        self.weights = (1.0,1.0,1.0,1.0,1.0)
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
