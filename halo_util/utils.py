import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)

def pad34_to_44(mat):
    last_row = torch.tensor([0., 0., 0., 1.], device=mat.device).reshape(1, 4).repeat(*mat.shape[:-2], 1, 1)
    return torch.cat([mat, last_row], dim=-2)

def convert_joints(joints, source, target):
    halo_joint_to_mano = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    mano_joint_to_halo = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])
    mano_joint_to_biomech = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
    biomech_joint_to_mano = np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])
    halo_joint_to_biomech = np.array([0, 13, 1, 4, 10, 7, 14, 2, 5, 11, 8, 15, 3, 6, 12, 9, 16, 17, 18, 19, 20])
    biomech_joint_to_halo = np.array([0, 2, 7, 12, 3, 8, 13, 5, 10, 15, 4, 9, 14, 1, 6, 11, 16, 17, 18, 19, 20])

    if source == 'halo' and target == 'biomech':

        return joints[:, halo_joint_to_biomech]
    if source == 'biomech' and target == 'halo':
        return joints[:, biomech_joint_to_halo]
    if source == 'mano' and target == 'biomech':
        return joints[:, mano_joint_to_biomech]
    if source == 'biomech' and target == 'mano':
        return joints[:, biomech_joint_to_mano]
    if source == 'halo' and target == 'mano':
        return joints[:, halo_joint_to_mano]
    if source == 'mano' and target == 'halo':
        return joints[:, mano_joint_to_halo]

    print("-- Undefined convertion. Return original tensor --")
    return joints
    
def change_axes(keypoints, source='mano', target='halo'):
    """Swap axes to match that of NASA
    """
    # Swap axes from local_cs to NASA
    kps_halo = keypoints + 0
    kps_halo[..., 0] = keypoints[..., 1]
    kps_halo[..., 1] = keypoints[..., 2]
    kps_halo[..., 2] = keypoints[..., 0]

    mat = torch.zeros([4, 4], device=keypoints.device)
    mat[0, 1] = 1.
    mat[1, 2] = 1.
    mat[2, 0] = 1.
    mat[3, 3] = 1.

    return kps_halo, mat


