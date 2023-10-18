import os
import tqdm
import os.path as osp
import numpy as np
import json
import torch
import pickle
import torch.nn.functional as F
from pytorch3d.transforms import rotation_conversions as rcvt

def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019

    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.

    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

# root = '/home/skynet/Zhifan/homan-master/results/epic_visor2_nobeta/samples'
root = '/home/skynet/Zhifan/homan-master/results/epic_visor2/step2/samples'
videos = os.listdir(root)
results = {}

degs = []
ds = []
for video in tqdm.tqdm(videos):
    f = osp.join(root, video, 'joint_fit.pt')
    if not os.path.exists(f):
        continue
    fit = torch.load(f)

    sd = fit['state_dict']

    To = sd['translations_object']
    Ro = sd['rotations_object']
    Ro = rot6d_to_matrix(Ro)

    Th = sd['translations_hand']
    Rh = sd['rotations_hand']
    Rh = rot6d_to_matrix(Rh)

    Ro2h = Ro.matmul(Rh.permute(0, 2, 1))
    To2h = (To - Th).matmul(Rh.permute(0, 2, 1))

    rd = Ro2h[0, ...] @ Ro2h[-1, ...].T
    rad = rcvt.matrix_to_axis_angle(rd).norm()
    deg = rad / np.pi * 180

    # Max translation change
    d = To2h[0, ...] - To2h[-1, ...]
    d = d.norm()

    deg = deg.item()
    d = d.item()
    degs.append(deg)
    ds.append(d)
    results[video] = {'deg': deg, 'd': d}

print(np.mean(degs))
print(np.mean(ds))

with open('report/epic_visor2_changes.json', 'w') as fp:
    json.dump(results, fp)