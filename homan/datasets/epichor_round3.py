#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
import cv2
import numpy as np
import torch

from homan.datasets import collate
from homan.tracking import trackhoa as trackhoadf
from homan.utils import bbox as bboxutils

import os
import os.path as osp
import pandas as pd
from PIL import Image
import trimesh
from libyana.lib3d import kcrop
from libyana.transformutils import handutils
from manopth import manolayer
from homan.datasets.epichor_reader_lib.reader import Reader


EPIC_HOA_SIZE = (1920, 1080)
VISOR_SIZE = (854, 480)


def keep_valid_frames(start: int,
                      end: int,
                      avail_frames: List,
                      num_frames:int) -> List:
    """ as uniform as possible:
    Create probes at uniform stops,
    for each probe, find closest unused unfiltered frame.

    Args:
        avail_frames: available absolute frame numbers
    """
    import bisect
    from copy import deepcopy
    probes = np.linspace(start, end, num_frames, endpoint=True, dtype=int)
    avail = [-np.inf] + deepcopy(avail_frames) + [np.inf]
    keep_frames = []
    for p in probes:
        if len(avail) == 2:
            break
        i = bisect.bisect_left(avail, p)
        l, r = avail[i-1], avail[i]
        if p-l < r-p:
            keep_frames.append(l)
            avail.pop(i-1)
        else:
            keep_frames.append(r)
            avail.pop(i)
    return keep_frames


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


def load_models():
    OBJ_ROOT = "/home/barry/Zhifan/epic_hor_method/weights/obj_models/epichor_export/"
    model_names = {'bottle', 'bowl', 'plate', 'can', 'cup', 'mug', 'glass', 'pan', 'saucepan'}

    models = {}
    for obj_name in model_names:
        obj_path = osp.join(OBJ_ROOT, f"{obj_name}.obj")
        obj = trimesh.load(obj_path)
        verts = np.asarray(obj.vertices, dtype=np.float32)
        models[obj_name] = {
            "verts": verts,
            "faces": np.array(obj.faces),
            "path": obj_path,
        }
    return models


class EPICHOR_ROUND3:
    def __init__(
        self,
        frame_nb=30,
        image_sets='/home/barry/Zhifan/epic_hor_method/code_epichor/image_sets/epichor_round3_2447valid_nonempty.csv',
        cache_dir='/media/barry/DATA/Zhifan/epic_hor_data/cache',
    ):
        """
        Arguments:
            track_padding (int): Number of frames to include before and after extent of action clip
                provided by the annotations
            frame_step (int): Number of frames to skip between two selected images
            verbs (list): subset of action verbs to consider
            nouns (list): subset of action verbs to consider
        """
        super().__init__()
        self.name = "epichor"
        self.object_models = load_models()

        self.vid_index = self.read_vid_index(image_sets)
        self.reader = Reader(mask_version='unfiltered')

        cache_fmt = osp.join(cache_dir, f'%s')
        self.hoa_hbox = torch.load(cache_fmt % 'hoa_hbox.pth')
        self.avail_mask_frames = torch.load(cache_fmt % 'avail_unfiltered_frames.pth')
        self.oboxes_cache = torch.load(cache_fmt % 'unfiltered_oboxes.pth')

        self.resize_factor = 1  # 3
        self.frame_nb = frame_nb
        self.image_size = (640, 360)  # == IMAGE_SIZE
        cache_folder = os.path.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)

        self.left_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                         side="left").th_faces.numpy()
        self.right_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                          side="right").th_faces.numpy()

    def read_vid_index(self, image_sets):
        infos = pd.read_csv(image_sets)
        infos['side'] = infos['handside'].str.replace(' hand', '')
        infos['start'] = infos['st']
        infos['end'] = infos['ed']
        return infos

    def get_roi(self, hbox, obox, res=640):
        """
        Get square ROI in xyxy format

        Args:
            hbox: (4,) xyxy
            obox: (4,) xyxy

        Returns:
            roi_bbox: (4,) xyxy abs
        """
        # Get all 2d points and extract bounding box with given image ratio
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
        info = self.vid_index.iloc[idx]
        frame_nb = self.frame_nb

        vid, cat, hos_name, long_side, start, end = \
            info.vid, info['cat'], info.hos_name, info.handside, info.st, info.ed
        mp4_name = info.mp4_name
        short_side = long_side.replace(' hand', '')
        assert long_side in {'left hand', 'right hand'}

        avail_frames = self.avail_mask_frames[mp4_name]
        avail_frames = [v for v in avail_frames if v in self.hoa_hbox[mp4_name]]
        frame_idxs = keep_valid_frames(start, end, avail_frames, frame_nb)

        hoa_box_scale = np.asarray(self.image_size * 2) / (EPIC_HOA_SIZE * 2)
        obj_box_scale = np.asarray(self.image_size * 2) / (VISOR_SIZE * 2)

        # Read images from tar file
        images = []
        masks_hand, masks_obj = [], []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        # for frame_id in frame_ids:
        for frame in frame_idxs:
            img = self.reader.read_image(vid, frame)
            # img = cv2.imread(self.image_fmt % (folder, video_id, frame_idx))
            # img = cv2.resize(img, self.image_size)
            # img = Image.fromarray(img[:, :, ::-1])
            img = self.reader.read_image_pil(vid, frame)  # No need to do BGR->RGB (?)
            img = img.resize(self.image_size)

            hand_box = self.hoa_hbox[mp4_name][frame]
            obj_box = self.oboxes_cache[mp4_name][frame]
            """ Convert xywh to xyxy """
            hand_box[2:] += hand_box[:2]
            obj_box[2:] += obj_box[:2]
            hand_box = hand_box * hoa_box_scale
            obj_bbox = obj_box * obj_box_scale

            roi, affine_trans = self.get_roi(hand_box, obj_bbox)
            img = handutils.transform_img(img, affine_trans, [res, res])
            images.append(img)

            # Occlusion ignore will be added later in the method code
            mask_pil = self.reader.read_mask_pil(vid, frame)
            _, mapping = self.reader.read_mask(vid, frame, return_mapping=True)
            mask_pil = mask_pil.resize(self.image_size, Image.NEAREST)
            mask_pil = handutils.transform_img(mask_pil, affine_trans, [res, res])
            mask = np.asarray(mask_pil)
            # mask = mask.astype(np.int32)

            mask_hand = np.zeros_like(mask)
            mask_hand[mask == mapping[long_side]] = 1
            mask_obj = np.zeros_like(mask)
            mask_obj[mask == mapping[hos_name]] = 1
            masks_hand.append(mask_hand)
            masks_obj.append(mask_obj)

            setup = {'object': 1, long_side: 1}
            obox = apply_bbox_transform(obj_bbox, affine_trans)
            canverts3d = self.object_models[cat]['verts']
            obj_faces = self.object_models[cat]['faces']
            obj_info = dict(
                verts3d=np.zeros_like(canverts3d), 
                canverts3d=canverts3d,
                faces=obj_faces,  
                path="NONE",
                bbox=obox.astype(np.float32))
            hbox = apply_bbox_transform(hand_box, affine_trans)
            hand_label = long_side.replace(' ', '_')   # "left hand" -> "left_hand"
            hand_info = dict(
                verts3d=np.zeros([778, 3]), 
                faces=self.left_faces if long_side == 'left hand' else self.right_faces,
                label=hand_label,
                bbox=hbox.astype(np.float32)
            )
            hand_infos = [hand_info]

            K = self.get_camintr()
            K = kcrop.get_K_crop_resize(
                torch.Tensor(K).unsqueeze(0), torch.Tensor([roi]),
                [res])[0].numpy()
            camera = dict(
                resolution=[res, res],  # WH
                K=K.astype(np.float32),
            )

            seq_obj_info.append(obj_info)
            seq_hand_infos.append(hand_infos)
            seq_cameras.append(camera)

        hand_nb = len(seq_hand_infos[0])
        collated_hand_infos = []
        for hand_idx in range(hand_nb):
            collated_hand_info = collate.collate(
                [hand[hand_idx] for hand in seq_hand_infos])
            collated_hand_info['label'] = collated_hand_info['label'][0]
            collated_hand_infos.append(collated_hand_info)

        annot_full_key = f"{cat}/{vid}_{short_side}_hand_{start}_{end}"
        obs = dict(hands=collated_hand_infos,
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=setup,
                   frame_idxs=frame_idxs,
                   images=images,
                   masks_hand=masks_hand,
                   masks_obj=masks_obj,
                   annot_full_key=annot_full_key)

        return obs

    def get_camintr(self):
        focal = 200
        cam_intr = np.array([
            [focal, 0, 640 // 2],
            [0, focal, 360 // 2],
            [0, 0, 1],
        ])
        return cam_intr

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

if __name__ == '__main__':
    # debug K
    ds = EPICHOR_ROUND3(30)
    annots = ds[0]
    print(annots)
