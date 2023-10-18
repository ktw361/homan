""" March 04 """
import numpy as np
import torch
from homan.homan import HOMan
from homan.interactions import scenesdf
from libyana.metrics.iou import batch_mask_iou

import sys
sys.path.append('/home/skynet/Zhifan/repos/CPF')
from hocontact.utils.libmesh.inside_mesh import check_mesh_contains
from libzhifan.geometry import SimpleMesh, visualize_mesh


def get_meshes(homan, scene_idx=0):
    hand_scale = homan.int_scales_hand
    hand_color = 'light_blue'
    obj_color = 'yellow'
    with torch.no_grad():
        verts_hand  = homan.get_verts_hand()[0] / hand_scale
        verts_obj = homan.get_verts_object()[0] / hand_scale
    f_hand = homan.faces_hand[0]
    f_obj = homan.faces_object[0]

    mhand = SimpleMesh(
        verts_hand[scene_idx], f_hand, tex_color=hand_color)
    mobj = SimpleMesh(
        verts_obj[scene_idx], f_obj, tex_color=obj_color)
    return mhand, mobj
    # return visualize_mesh([mhand, mobj], show_axis=False, viewpoint='nr')


def to_scene(homan, scene_idx=-1, show_axis=False, viewpoint='nr'):
    """
    homan.faces_hand: (1, 1538, 3)
    homan.faces_object: (30, 2000, 3)
    homan.get_verts_hand()[0] (30, 778, 3)
    homan.get_verts_object()[0] (30, 1000, 3)
    """
    with torch.no_grad():
        verts_hand, _  = homan.get_verts_hand()
    T = len(verts_hand)

    if scene_idx >= 0:
        mhand, mobj = get_meshes(homan, scene_idx)
        return visualize_mesh([mhand, mobj], show_axis=show_axis, viewpoint=viewpoint)

    meshes = []
    disp = 0.15  # displacement
    T = len(verts_hand)
    for t in range(T):
        mhand, mobj = get_meshes(homan, t)
        mhand.apply_translation_([t * disp, 0, 0])
        mobj.apply_translation_([t * disp, 0, 0])
        meshes.append(mhand)
        meshes.append(mobj)
    return visualize_mesh(meshes, show_axis=show_axis, viewpoint=viewpoint)


def compute_pen_depth(homan, h2o_only=True) -> float:
    """ report in mm ,
    max penetration depth in the whole sequence
    """
    hand_scale = homan.int_scales_hand
    # inv_scale_vol = (1 / hand_scale)**3
    f_hand = homan.faces_hand[0]
    f_obj = homan.faces_object[0]
    v_hand = homan.get_verts_hand()[0] / hand_scale
    v_obj = homan.get_verts_object()[0] / hand_scale

    sdfl = scenesdf.SDFSceneLoss([f_hand, f_obj])
    _, sdf_meta = sdfl([v_hand, v_obj])
    h_to_o = sdf_meta['dist_values'][(1, 0)].max(1)[0].max().item() 
    o_to_h = sdf_meta['dist_values'][(0, 1)].max(1)[0].max().item() 
    
    h_to_o = h_to_o * 1000
    o_to_h = o_to_h * 1000
    if not h2o_only:
        return h_to_o, o_to_h
    else:
        return h_to_o


def max_intersect_volume(homan, pitch=0.005, ret_all=False):
    """ Iv of object into hand, report in cm^3
    pitch: voxel size, 0.01m == 1cm
    """
    with torch.no_grad():
        verts_hand = homan.get_verts_hand()[0]

    max_iv = 0
    T = len(verts_hand)
    vox_size = np.power(pitch * 100, 3)
    iv_list = []
    for t in range(T):
        mhand, mobj = get_meshes(homan, t)

        obj_pts = mobj.voxelized(pitch=pitch).points
        inside = check_mesh_contains(mhand, obj_pts)
        volume = inside.sum() * vox_size
        # inside = mhand.contains(obj_points)
        # volume = inside.sum() * vox_size
        iv_list.append(volume)
        max_iv = max(max_iv, volume)
    if ret_all:
        return iv_list
    else:
        return max_iv

def compute_obj_iou(homan: HOMan):
    v_obj = homan.get_verts_object()[0]
    f_obj = homan.faces_object.expand(len(v_obj), -1, -1)
    return homan.losses.compute_sil_loss_object(v_obj, f_obj)[1]['iou_object']


def compute_hand_iou(homan: HOMan):
    v_hand = homan.get_verts_hand()[0]
    f_hand = homan.faces_hand.expand(len(v_hand), -1, -1)
    # Rendering happens in ROI
    camintr = homan.camintr_rois_hand
    rend = homan.losses.renderer(v_hand, f_hand, K=camintr, mode="silhouettes")
    image = homan.keep_mask_hand * rend
    ious = batch_mask_iou(image, homan.ref_mask_hand)
    return ious.mean().item()