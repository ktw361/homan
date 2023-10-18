#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from homan.datasets import collate, epichoa, tarutils
from homan.utils import bbox as bboxutils

import pandas as pd
import trimesh
from libyana.lib3d import kcrop
from libyana.transformutils import handutils
from manopth import manolayer
from homan.datasets.arctic_lib.data_reader_onthefly import SeqReaderOnTheFly
from libzhifan.geometry import CameraManager
from homan.datasets.arctic_lib.data_reader import (
    CROPPED_IMAGE_SIZE,
)

LEFT = 'left'
RIGHT = 'right'


def apply_bbox_transform(bbox, affine_trans):
    x_min, y_min = handutils.transform_coords(
        [bbox[:2]],
        affine_trans,
    )[0]
    x_max, y_max = handutils.transform_coords(
        [bbox[2:]],
        affine_trans,
    )[0]
    new_bbox = np.array([x_min, y_min, x_max, y_max])
    return new_bbox


def load_models(MODELS, normalize=True):
    models = {}
    for obj_name, obj_info in MODELS.items():
        obj_path = obj_info["path"]
        scale = obj_info["scale"]
        obj = trimesh.load(obj_path)
        verts = np.array(obj.vertices)
        if normalize:
            # center
            verts = verts - verts.mean(0)
            # inscribe in 1-radius sphere
            verts = verts / np.linalg.norm(verts, 2, 1).max() * scale / 2
        models[obj_name] = {
            "verts": verts,
            "faces": np.array(obj.faces),
            "path": obj_path,
        }
    return models



class ArcticStable:
    def __init__(
        self,
        frame_nb=30,
    ):
        super().__init__()
        self.name = "arctic_stable"
        self.frame_nb = frame_nb
        self.image_size = (640, 360)  # == IMAGE_SIZE

        self.box_factor = np.asarray(self.image_size * 2) / (CROPPED_IMAGE_SIZE * 2)
        self.left_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                         side="left").th_faces.numpy()
        self.right_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                          side="right").th_faces.numpy()
        # self.faces = {"left": left_faces, "right": right_faces}

        self.vid_index = pd.read_csv(
            './local_data/arctic_outputs/stable_grasps_v3_frag_valid_min20.csv')

    def get_roi(self, hbox, obox, res=640):
        """
        Get square ROI in xyxy format

        Args:
            hbox: (4,) xywh
            obox: (4,) xywh

        Returns:
            roi_bbox: (4,) xyxy abs
        """
        # Get all 2d points and extract bounding box with given image
        # ratio
        bboxes = [hbox, obox]
        # bboxes = [bboxs[frame_ids] for bboxs in annots["bboxes_xyxy"].values()]
        # all_vid_points = np.concatenate(list(bboxes)) / self.resize_factor
        all_vid_points = np.vstack(bboxes) #/ self.box_factor
        xy_points = np.concatenate(
            [all_vid_points[:, :2], all_vid_points[:, 2:]], 0)
        mins = xy_points.min(0)
        maxs = xy_points.max(0)
        roi_box_raw = np.array([mins[0], mins[1], maxs[0], maxs[1]])
        roi_bbox = bboxutils.bbox_wh_to_xy(
            bboxutils.make_bbox_square(bboxutils.bbox_xy_to_wh(roi_box_raw),
                                       bbox_expansion=0.2))
        roi_center = (roi_bbox[:2] + roi_bbox[2:]) / 2
        # Assumes square bbox
        roi_scale = roi_bbox[2] - roi_bbox[0]
        affine_trans = handutils.get_affine_transform(roi_center, roi_scale,
                                                      [res, res])[0]
        return roi_bbox, affine_trans

    def __getitem__(self, idx, res=640):
        vid_info = self.vid_index.iloc[idx]

        start, end = vid_info.start, vid_info.end
        sid_seq_name = vid_info.sid_seq_name
        sid, seq_name = sid_seq_name.split('/')
        cat = seq_name.split('_')[0]
        side = vid_info['side']
        frame_idxs = np.linspace(start, end, num=self.frame_nb, dtype=int, endpoint=True)  # [S, S+1, ... , E]
        seq_reader = SeqReaderOnTheFly(sid, seq_name, obj_name=cat, obj_version='reduced')

        v_obj, f_obj, T_o2l, T_o2r = seq_reader.neutralized_obj_params()  # (N, V, 3) (F, 3), (N, 4, 4), (N, 4, 4) GT
        v_obj = v_obj[start]  # (V, 3)
        T_o2h = T_o2l if side == LEFT else T_o2r
        T_o2h = T_o2h[frame_idxs]  # (T, 4, 4)

        vo_cam = seq_reader.obj_verts(None, space='ego')  # (N, V, 3)
        vh = seq_reader.hand_verts(None, side=side, space='ego')  # (N, V, 3)

        # Read images from tar file
        images = []
        masks_hand, masks_obj = [], []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []

        gt_person_parameters = []
        # roi, affine_trans = self.get_roi(vid_info)
        for frame_idx in frame_idxs:
            img = seq_reader.render_image(frame_idx)
            img = cv2.resize(img, self.image_size)
            img = Image.fromarray(img[:, :, ::-1])

            mask, lbox, rbox, obox = seq_reader.get_boxes_and_mask(frame_idx, use_disk=True)
            obox[2:] += obox[:2]
            obox = obox * self.box_factor

            hbox = lbox if side == LEFT else rbox
            hbox[2:] += hbox[:2]
            hbox = hbox * self.box_factor

            roi, affine_trans = self.get_roi(hbox, obox)
            img = handutils.transform_img(img, affine_trans, [res, res])
            images.append(img)

            mask = Image.fromarray(mask)
            mask = mask.resize(self.image_size, Image.NEAREST)
            mask = handutils.transform_img(mask, affine_trans, [res, res])
            mask = np.asarray(mask)
            mask_hand = mask == (1 if side == LEFT else 2)
            mask_obj = mask == 3
            masks_hand.append(mask_hand)
            masks_obj.append(mask_obj)

            setup = {'objects': 1,
                'left_hand' if side == LEFT else 'right_hand': 1, }
            obox = apply_bbox_transform(obox, affine_trans)
            obj_info = dict(
                verts3d=vo_cam[frame_idx],  # This should be GT
                faces=f_obj,
                path="NONE",
                canverts3d=v_obj,  # This should be initial
                bbox=obox.astype(np.float32))  # xyxy

            hbox = apply_bbox_transform(hbox, affine_trans)
            hand_info = dict(
                verts3d=vh[frame_idx],
                faces=self.left_faces if side == LEFT else self.right_faces,
                label="left_hand" if side == LEFT else 'right_hand',
                bbox=hbox.astype(np.float32)
            )
            hand_infos = [hand_info]

            K = self.get_camintr(seq_reader)  # K should be global K here
            K = kcrop.get_K_crop_resize(
                torch.Tensor(K).unsqueeze(0), torch.Tensor([roi]),
                [res])[0].numpy()
            camera = dict(
                resolution=[res, res],  # WH
                K=K.astype(np.float32),
            )

            person_param = self.gt_person_parameter(
                seq_reader, frame_idx, side, hbox, K)

            seq_obj_info.append(obj_info)
            seq_hand_infos.append(hand_infos)
            seq_cameras.append(camera)
            gt_person_parameters.append(person_param)

        hand_nb = len(seq_hand_infos[0])
        collated_hand_infos = []
        for hand_idx in range(hand_nb):
            collated_hand_info = collate.collate(
                [hand[hand_idx] for hand in seq_hand_infos])
            collated_hand_info['label'] = collated_hand_info['label'][0]
            collated_hand_infos.append(collated_hand_info)

        T_h2e = seq_reader.pose_hand2ego(side)
        R_o2h_prior, t_o2h_prior = self.get_T_o2h_priors(side, cat)  # learnt priors
        T_o2h_prior = torch.eye(4).repeat(R_o2h_prior.shape[0], 1, 1)
        T_o2h_prior[:, :3, :3] = R_o2h_prior
        T_o2h_prior[:, :3, 3] = t_o2h_prior.view(1, 3).repeat(R_o2h_prior.shape[0], 1)
        T_h2e_prior = T_h2e[[start]].matmul(T_o2h_prior)  # Yana uses first frame
        R_o2e_prior = T_h2e_prior[:, :3, :3].permute(0, 2 ,1)  # (num_inits, 3, 3)
        t_o2e_prior = T_h2e_prior[:, :3, 3]  # (num_inits, 3, 3)

        # GT obj-to-ego, for debug use
        T_o2e = T_h2e[frame_idxs].matmul(T_o2h)
        gt_R_o2e = T_o2e[:, :3, :3].permute(0, 2, 1)
        gt_t_o2e = T_o2e[:, :3, 3]
        # debug 
        # R_o2e_prior[0, :3, :3] = gt_R_o2e[0, :3, :3]
        # t_o2e_prior[0, :3] = gt_t_o2e[0, :3]
        diameter = seq_reader.obj_diameter

        annot_full_key = "%s_%d_%d_%s" % (
            sid_seq_name, start, end, side)
        obs = dict(hands=collated_hand_infos,
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=setup,
                   frame_idxs=frame_idxs,
                   images=images,
                   masks_hand=masks_hand,
                   masks_obj=masks_obj,
                   annot_full_key=annot_full_key,
                   gt_person_parameters=gt_person_parameters,
                   rotations_inits=R_o2e_prior,
                   translations_inits=t_o2e_prior,
                   gt_R_o2e=gt_R_o2e,
                   gt_t_o2e=gt_t_o2e,
                   diameter=diameter,
        )

        return obs

    def gt_person_parameter(self,
                            seq_reader: SeqReaderOnTheFly,
                            f,
                            side,
                            hbox,
                            camintr):
        """
        Args:
            f: frame index
        """
        device = 'cpu'

        T_h2e = seq_reader.pose_hand2ego(side)  # (N, 4, 4)
        rot_h2e = T_h2e[:, :3, :3]
        # gt_pose_pca = seq_reader.pose_mano(side, is_pca=True)
        gt_pose = seq_reader.pose_mano(side, is_pca=False)[[f]]
        gt_hand_betas = torch.from_numpy(seq_reader.mano_params[side]['shape']).view(1, 10)
        mano_layer_side = seq_reader.l_mano_layer if side == LEFT else seq_reader.r_mano_layer
        gt_pose_pca = recover_pca_pose(gt_pose, mano_layer_side).view(1, 45)

        verts = seq_reader.hand_verts(f, side, space=side, as_mesh=False, zero_rot_trans=False)  # (778, 3), untransformed
        verts = verts.view(1, 778, 3).to(device)
        faces = seq_reader.fl if side == LEFT else seq_reader.fr
        faces = torch.as_tensor(faces, device=device).view(1, 1538, 3)
        rotations = rot_h2e[f].view(1, 3, 3).to(device)
        rotations = rotations.permute(0, 2, 1)  # in HOMan it's V @ R
        translations = T_h2e[f, :3, 3].view(1, 1, 3).to(device)
        mano_pose = gt_pose.view(1, 45).to(device)
        mano_rot = mano_pose.new_zeros(1, 3)
        mano_trans = mano_pose.new_zeros(1, 3)
        verts2d = torch.as_tensor(camintr).matmul(verts[0].permute(1, 0)).permute(1, 0)
        verts2d = verts2d / verts2d[:, 2:]
        verts2d = verts2d[:, :2].view(1, 778, 2)
        verts2d = torch.zeros_like(verts2d)  # Just use GT hand. (verts2d is called in loss_2d, but just fix to GT for now)

        person_param = dict(
            bboxes=torch.as_tensor(hbox, device=device).view(1, 4),
            faces=faces,
            verts=verts,  # GT ego verts in 3D
            verts2d=verts2d,
            rotations=rotations,
            mano_pose=mano_pose,
            mano_pca_pose=gt_pose_pca,
            mano_rot=mano_rot,
            mano_betas=gt_hand_betas,
            mano_trans=mano_trans,
            translations=translations,
            hand_side=[side],
        )
            # 'bboxes', 'faces', 'verts', 'verts2d', 'rotations', 'mano_pose', 'mano_pca_pose', 'mano_rot', 'mano_betas', 'mano_trans', 'translations', 'hand_side',
            # 'masks'])
            # person_parameters[i] = person_param
        return person_param

    def get_camintr(self, seq_reader: SeqReaderOnTheFly):
        """ global camera, do not normalise  """
        K_ego = seq_reader.K_ego[0]
        # low_h = CROPPED_IMAGE_SIZE[1]
        # low_w = CROPPED_IMAGE_SIZE[0]
        low_w, low_h = self.image_size
        orig_h = 2000
        orig_w = 2800
        w_ratio = low_w / orig_w
        h_ratio = low_h / orig_h
        fx = K_ego[0, 0] * w_ratio
        fy = K_ego[1, 1] * h_ratio
        cx = K_ego[0, 2] * w_ratio
        cy = K_ego[1, 2] * h_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=low_h, img_w=low_w)
        cam_intr = cam_manager.get_K()
        # focal = 200
        # cam_intr = np.array([
        #     [focal, 0, 640 // 2],
        #     [0, focal, 360 // 2],
        #     [0, 0, 1],
        # ])
        return cam_intr

    def get_T_o2h_priors(self, side, cat):
        """
        Returns:
            R: (num_priors = 50, 3, 3) torch.Tensor
            t: (3,) torch.Tensor
        """
        arctic_cat = f'arctic_{cat}'
        neutral_priors = torch.load('local_data/weights/pose_priors/neutral_priors.pth')
        priors = neutral_priors[arctic_cat]
        if side == LEFT:
            R_o2h = priors['R_o2l']
            t_o2h = priors['t_o2l']
        else:
            R_o2h = priors['R_o2r']
            t_o2h = priors['t_o2r']
        return R_o2h, t_o2h

    # def get_focal_nc(self):
    #     cam_intr = self.get_camintr()
    #     return (cam_intr[0, 0] + cam_intr[1, 1]) / 2 / max(self.image_size)

    def __len__(self):
        return len(self.vid_index)

    def project(self, points3d, cam_intr, camextr=None):
        if camextr is not None:
            points3d = np.array(self.camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)


def recover_pca_pose(pred_hand_pose: torch.Tensor, mano_layer_side) -> torch.Tensor:
    """
    if
        v_exp = ManopthWrapper(pca=False, flat=False).(x_0)
        x_pca = self.recover_pca_pose(self.x_0)  # R^45
    then
        v_act = ManoLayer(pca=True, flat=False, ncomps=45).forward(x_pca)
        v_exp == v_act

    note above requires mano_rot == zeros, since the computation of rotation
        is different in ManopthWrapper
    """
    M_pca_inv = torch.inverse(mano_layer_side.th_comps)
    mano_pca_pose = pred_hand_pose.mm(M_pca_inv)
    return mano_pca_pose


if __name__ == '__main__':
    # debug K
    ds = ArcticStable(30)
    annots = ds[0]
