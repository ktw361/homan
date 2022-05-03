#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from homan.datasets import collate, epichoa, epic_hdf5
from homan.datasets.chunkvids import chunk_vid_index
from homan.tracking import trackhoa as trackhoadf
from homan.utils import bbox as bboxutils

import pandas as pd
import pickle
import trimesh
from libyana.lib3d import kcrop
from libyana.transformutils import handutils
from manopth import manolayer

SHAPENET_PATH = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2/"
"""
Shapenet list: https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e
"""

MODELS = {
    "bottle": {
        "path": "data/cache/models/bottle.obj",
        # "path": SHAPENET_PATH+
        # "02876657/d851cbc873de1c4d3b6eb309177a6753/models/model_normalized_proc.obj",
        "scale": 0.2,
    },
    # "jug": {
    #     "path": "data/cache/models/jug.obj",
        # "local_data/datasets/ho3dv2/processmodels/019_pitcher_base/textured_simple_400.obj",
        # '/media/eve/DATA/Zhifan/YCB_Video_Models/models/019_pitcher_base/textured_simple.obj',
    #     "scale": 0.25,
    # },
    # "pitcher": {
    #     "path":
    #     # "local_data/datasets/ho3dv2/processmodels/019_pitcher_base/textured_simple_400.obj",
    #     '/media/eve/DATA/Zhifan/YCB_Video_Models/models/019_pitcher_base/textured_simple.obj',
    #     "scale": 0.25,
    # },
    "plate": {
        "path": "data/cache/models/plate.obj",
        # "path": SHAPENET_PATH+
        # "02880940/95ac294f47fd7d87e0b49f27ced29e3/models/model_normalized_proc.obj",
        "scale": 0.3,
    },
    "cup": {
        "path": "data/cache/models/cup.obj",
        # "path": SHAPENET_PATH+
        # "03797390/d75af64aa166c24eacbe2257d0988c9c/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "phone": {
        "path": "data/cache/models/phone.obj",
        # "path": SHAPENET_PATH+
        # "02992529/7ea27ed05044031a6fe19ebe291582/models/model_normalized_proc.obj",
        "scale": 0.07
    },
    "can": {
        "path": "data/cache/models/can.obj",
        # "path": SHAPENET_PATH+
        # "02946921/3fd8dae962fa3cc726df885e47f82f16/models/model_normalized_proc.obj",
        "scale": 0.2
    },
    "mug": {
        "path": "data/cache/models/can.obj",
        "scale": 0.12,
    },
    "bowl": {
        "path": "data/cache/models/bowl.obj",
        "scale": 0.12,
    }

}


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
        if obj_path.startswith('http'):
            obj = trimesh.load_remote(
                    obj_path.replace('_proc',''), force='mesh')
        else:
            obj = trimesh.load(obj_path, force='mesh')
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


class Epic:

    """
    Structure of 
        - self.vid_index: DataFrame
                    seq_idx              | frame_nb | start_frame | object | verb
            eg. (P01_01, 0, P01_01_100)  | 45       | 28802       | cup    | take

            where the 'seq_idx' = (video_id, annot_idx, narration_id)

        - self.annotations: dict, len == len(self.vid_index)
            - key:  seq_idx of self.vid_index
            - value: dict with
                - 'bboxes_xyxy':  dict with
                    - 'objects':    ndarray (frame_nb, 4)
                    - 'right_hand': ndarray (frame_nb, 4)
                - 'frame_idxs': list of len frame_nb
                    [start_frame, start_frame + 1, ..., start_frame + frame_nb - 1]
        
        - self.chunk_index: (example: chunk_size=10, chunk_step=1)
                    seq_idx              | frame_nb | start_frame | object | verb | frame_idxs
            eg. (P01_01, 0, P01_01_100)  | 45       | 28802       | cup    | take | [0, 1, ..., 9]
                (P01_01, 0, P01_01_100)  | 45       | 28802       | cup    | take | [10, 11, ..., 19]

    
    Loading example:
    Originally, len(annot_df) = 67217;
    after filter valid segmentation,    len(annot_df) = 9022
    after filter with nouns,            len(annot_df) = 301     <- annot_idx
    after computer_tracks(), where short tracks are removed
                                        len(annot_df) = 257
    after chunking (split tracks into smaller sequences),
                                        len(chunk_index) = 2799
    """

    def __init__(
        self,
        joint_nb=21,
        use_cache=False,
        hdf5_root=None,
        mano_root="extra_data/mano",
        mode="frame",
        ref_idx=0,
        valid_step=-1,
        frame_step=1,
        frame_nb=10,
        verbs=[  # "take", "put",
            "take", "put"
            "open", "close"
        ],
        nouns=[
            "can",
            "cup",
            "plate",
            "bottle",
            # "pitcher",
            # "jug",
            # "phone",
        ],
        box_folder="data/boxes",
        track_padding=10,
        min_frame_nb=20,
        epic_root="/home/skynet/Zhifan/datasets/epic", 
        valid_vids_path='/home/skynet/Zhifan/data/allVideos.xlsx',
    ):
        """
        Arguments:
            min_frame_nb (int): Only sequences with length at least min_frame_nb are considered
            track_padding (int): Number of frames to include before and after extent of action clip
                provided by the annotations
            frame_step (int): Number of frames to skip between two selected images
            verbs (list): subset of action verbs to consider
            nouns (list): subset of action verbs to consider
            valid_vids_path: str, the path to .xlsx file 
                that stores videos with segmentation annotation

        """
        super().__init__()
        self.name = "epic"
        self.mode = mode
        self.frame_step = frame_step
        self.nouns = nouns
        self.track_padding = track_padding
        self.min_frame_nb = min_frame_nb
        self.epic_root = epic_root
        self.valid_vids_path = valid_vids_path
        self.image_size = (640, 360)

        self.object_models = load_models(MODELS)
        if hdf5_root is not None:
            self.hdf5_reader = epic_hdf5.EpicHdf5Reader(hdf5_root)
            def _read_frame(video_id, frame_idx):
                img = self.hdf5_reader.read_frame_np(video_id, f"frame_{frame_idx:010d}")
                img = cv2.resize(img, self.image_size)
                img = Image.fromarray(img)
                return img
        else:
            self.frame_template = osp.join(epic_root, "rgb_root",
                                           "{}/{}/frame_{:010d}.jpg")
            def _read_frame(video_id, frame_idx):
                img_path = self.frame_template.format(
                        video_id[:3], video_id, frame_idx)
                img = cv2.imread(img_path)
                img = cv2.resize(img, self.image_size)
                img = Image.fromarray(img[:, :, ::-1])
                return img
        self._read_frame = _read_frame

        self.resize_factor = 3
        self.frame_nb = frame_nb
        self.frame_step = frame_step
        cache_folder = osp.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)

        left_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                         side="left").th_faces.numpy()
        right_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                          side="right").th_faces.numpy()
        self.faces = {"left": left_faces, "right": right_faces}

        self.cache_path = self._get_cache_path()
        if os.path.exists(self.cache_path) and use_cache:
            with open(self.cache_path, "rb") as p_f:
                dataset_annots = pickle.load(p_f)
            vid_index = dataset_annots["vid_index"]
            annotations = dataset_annots["annotations"]
        else:
            vid_index, annotations = self._init_tracks()
            dataset_annots = {
                "vid_index": vid_index,
                "annotations": annotations,
            }
            with open(self.cache_path, "wb") as p_f:
                pickle.dump(dataset_annots, p_f)

        self.annotations = annotations
        self.vid_index = vid_index

        self.chunk_index = self._init_chunk_vid_index(vid_index)

        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def _get_cache_path(self):
        return 'data/cache/epic_take_putopen_close_can_cup_phone_plate_pitcher_jug_bottle_20.pkl'

    def _init_tracks(self):
        with open(osp.join(self.epic_root, "EPIC_100_train.pkl"), "rb") as p_f:
            annot_df = pickle.load(p_f)
        valid_vids = pd.read_excel(self.valid_vids_path)

        # 1. Select video with gt segs
        annot_df = annot_df[annot_df.video_id.isin(valid_vids['Unnamed: 0'])]
        # 2. Select interested sub-sequence
        annot_df = annot_df[annot_df.noun.isin(self.nouns)]

        print(f"Processing {annot_df.shape[0]} clips for nouns {self.nouns}")
        vid_index, annotations = self.compute_tracks(annot_df)

        vid_index = pd.DataFrame(vid_index)
        return vid_index, annotations
    
    def _init_chunk_vid_index(self, vid_index):
        chunk_index = chunk_vid_index(vid_index,
                                      chunk_size=self.frame_nb,
                                      chunk_step=self.frame_step,
                                      chunk_spacing=self.frame_step * self.frame_nb)
        chunk_index = chunk_index[chunk_index.object.isin(
            self.nouns)]
        print(f"Working with {len(chunk_index)} chunks for {self.nouns}")
        return chunk_index
    
    def compute_tracks(self, annot_df):
        vid_index = []
        annotations = {}
        hoa_dets_cache = {}
        TOTAL = len(annot_df)
        # TOTAL = min(5, len(annot_df))
        with tqdm(total = TOTAL) as pbar:
            for annot_idx, (annot_key,
                            annot) in enumerate(annot_df.iterrows()):

                if annot.video_id not in hoa_dets_cache:
                    hoa_dets = epichoa.load_video_hoa(
                        annot.video_id,
                        hoa_root=osp.join(self.epic_root, "hoa"))
                    hoa_dets = hoa_dets[
                        hoa_dets.left < hoa_dets.right][
                            hoa_dets.top < hoa_dets.bottom]
                    hoa_dets_cache[annot.video_id] = hoa_dets
                else:
                    hoa_dets = hoa_dets_cache[annot.video_id]

                frame_idxs, bboxes = trackhoadf.track_hoa_df(
                    hoa_dets,
                    video_id=annot.video_id,
                    start_frame=max(1, annot.start_frame - self.track_padding),
                    end_frame=(min(annot.stop_frame + self.track_padding,
                                hoa_dets.frame.max() - 1)),
                    dt=self.frame_step / 60,
                )
                if len(frame_idxs) > self.min_frame_nb:
                    annot_full_key = (annot.video_id, annot_idx, annot_key)
                    vid_index.append({
                        "seq_idx": annot_full_key,
                        "frame_nb": len(frame_idxs),
                        "start_frame": min(frame_idxs),
                        "object": annot.noun,
                        "verb": annot.verb,
                    })
                    annotations[annot_full_key] = {
                        "bboxes_xyxy": bboxes,
                        "frame_idxs": frame_idxs
                    }
                else:
                    logging.info(
                        f"Skip {annot.video_id} with num_frames = "
                        f"{len(frame_idxs)} < {self.min_frame_nb}")
                
                pbar.update(1)
            # except Exception:
            #     print(f"Skipping idx {annot_idx}")
        return vid_index, annotations
        
    def get_roi(self, video_annots, frame_ids, res=640):
        """
        Get square ROI in xyxy format

        roi_expansion was 0.2
        """
        # Get all 2d points and extract bounding box with given image
        # ratio
        annots = self.annotations[video_annots.seq_idx]
        bboxes = [bboxs[frame_ids] for bboxs in annots["bboxes_xyxy"].values()]
        all_vid_points = np.concatenate(list(bboxes)) / self.resize_factor
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

    def __getitem__(self, idx):
        """
        Returns:

        - when self.mode == 'vid':
            ...

        - when self.mode == 'chunk':
            dict of 
                - 'hands': list of len 1(why?) dict of
                    - 'verts3d':    (N, 788, 3)
                    - 'faces':      (N, 1538, 3)
                    - 'label':      str, one of {'left_hand', 'right_hand'}
                    - 'bbox':       (N, 4)

                - 'objects': list of len 1 dict of
                    - 'verts3d':    (N, V_o, 3)
                    - 'faces':      (N, F_o, 3)
                    - 'path':       list of len N path, 
                        e.g. ['data/cache/models/cup.obj', ...]
                    - 'canverts3d': (N, V_o, 3)
                    - 'bbox':       (N, 4)

                - 'camera': dict of 
                    - 'resolution': list of len N list [rx, ry],
                        e.g. [[640, 640] * N]
                    - 'K':          (N, 3, 3)

                - 'setup': dict,
                    e.g. {'objects': 1, 'right_hand': 1}

                - 'frame_idxs': list of len N true frame index
                    e.g. [28802, 28803, ..., 28811]

                - 'images': list of len N PIL.Images
                - 'seq_idx': tuple of (video_id, annot_idx, narration_id)
            
            where N = self.frame_nb, (e.g. 10),
            all data are ndarray

        """
        if self.mode == "vid":
            return self.get_vid_info(idx, mode="full_vid")
        elif self.mode == "chunk":
            return self.get_vid_info(idx, mode="chunk")

    def get_vid_info(self, idx, res=640, mode="full_vid"):
        # Use all frames if frame_nb is -1
        if mode == "full_vid":
            vid_info = self.vid_index.iloc[idx]
            # Use all frames if frame_nb is -1
            if self.frame_nb == -1:
                frame_nb = vid_info.frame_nb
            else:
                frame_nb = self.frame_nb

            frame_ids = np.linspace(0, vid_info.frame_nb - 1,
                                    frame_nb).astype(np.int)
        else:
            vid_info = self.chunk_index.iloc[idx]
            frame_ids = vid_info.frame_idxs
        seq_frame_idxs = [self.annotations[vid_info.seq_idx]["frame_idxs"]][0]
        frame_idxs = [seq_frame_idxs[frame_id] for frame_id in frame_ids]
        video_id = vid_info.seq_idx[0]

        # Read images from tar file
        images = []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        roi, affine_trans = self.get_roi(vid_info, frame_ids)
        for frame_id in frame_ids:
            frame_idx = seq_frame_idxs[frame_id]
            img = self._read_frame(video_id, frame_idx)

            img = handutils.transform_img(img, affine_trans, [res, res])
            images.append(img)

            obj_info, hand_infos, camera, setup = self.get_hand_obj_info(
                vid_info,
                frame_id,
                roi=roi,
                res=res,
                affine_trans=affine_trans)
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

        obs = dict(hands=collated_hand_infos,
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=setup,
                   frame_idxs=frame_idxs,
                   images=images,
                   seq_idx=vid_info.seq_idx)

        return obs

    def get_hand_obj_info(self,
                          frame_info,
                          frame,
                          res=640,
                          roi=None,
                          affine_trans=None):
        hand_infos = []
        video_annots = self.annotations[frame_info.seq_idx]
        bbox_names = video_annots['bboxes_xyxy'].keys()
        bbox_infos = video_annots['bboxes_xyxy']
        setup = {"objects": 1}
        for bbox_name in bbox_names:
            setup[bbox_name] = 1
        has_left = "left_hand" in bbox_names
        has_right = "right_hand" in bbox_names

        if has_right:
            bbox = bbox_infos['right_hand'][frame] / self.resize_factor
            bbox = apply_bbox_transform(bbox, affine_trans)
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["right"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="right_hand",
                bbox=bbox.astype(np.float32),
            )
            hand_infos.append(hand_info)
        if has_left:
            bbox = bbox_infos['left_hand'][frame] / self.resize_factor
            bbox = apply_bbox_transform(bbox, affine_trans)
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["left"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="left_hand",
                bbox=bbox.astype(np.float32),
            )
            hand_infos.append(hand_info)

        K = self.get_camintr()
        K = kcrop.get_K_crop_resize(
            torch.Tensor(K).unsqueeze(0), torch.Tensor([roi]),
            [res])[0].numpy()
        obj_info = self.object_models[frame_info.object]
        obj_bbox = bbox_infos["objects"][frame] / self.resize_factor
        obj_bbox = apply_bbox_transform(obj_bbox, affine_trans)
        verts3d = obj_info["verts"] + np.array([0, 0, 0.6])

        obj_info = dict(verts3d=verts3d.astype(np.float32),
                        faces=obj_info['faces'],
                        path=obj_info['path'],
                        canverts3d=obj_info["verts"].astype(np.float32),
                        bbox=obj_bbox)
        camera = dict(
            resolution=[res, res],  # WH
            K=K.astype(np.float32),
        )
        return obj_info, hand_infos, camera, setup

    def get_camintr(self):
        focal = 200
        cam_intr = np.array([
            [focal, 0, 640 // 2],
            [0, focal, 360 // 2],
            [0, 0, 1],
        ])
        return cam_intr

    def get_focal_nc(self):
        cam_intr = self.get_camintr()
        return (cam_intr[0, 0] + cam_intr[1, 1]) / 2 / max(self.image_size)

    def __len__(self):
        if self.mode == "vid":
            return len(self.vid_index)
        elif self.mode == "chunk":
            return len(self.chunk_index)
        else:
            raise ValueError(f"{self.mode} mode not in [frame|vid|chunk]")

    def project(self, points3d, cam_intr, camextr=None):
        if camextr is not None:
            points3d = np.array(self.camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)


class EpicFrame(Epic):
    
    def __init__(
        self,
        frames_file,
        interpolation_dir='/home/skynet/Zhifan/data/epic_analysis/interpolation',
        *args,
        **kwargs):

        self._frames_file = frames_file
        self._interpolation_dir = interpolation_dir

        self.with_interp_mask = True
        if self._interpolation_dir is None:
            self.with_interp_mask = False
            
        super(EpicFrame, self).__init__(*args, **kwargs)

    def _get_cache_path(self):
        return 'data/cache/epic_frame.pkl'
    
    def _init_tracks(self):
        with open(self._frames_file) as fp:
            lines = fp.readlines()

        vid_index = []
        for i, line in enumerate(lines):
            line = line.strip().replace('\t', ' ')
            nid, cat, side, st_frame = [v for v in line.split(' ') if len(v) > 0]
            vid = '_'.join(nid.split('_')[:2])
            st_frame = int(st_frame)
            side = '_'.join([side, 'hand'])
            vid_index.append([
                (vid, i, nid), 1, st_frame, cat, 'none', side])
        vid_index = pd.DataFrame(
            vid_index, columns=['seq_idx', 'frame_nb', 
                                'start_frame', 'object', 'verb', 'side'])
        
        def _df2boxes(df):
            boxes = np.asarray([
                epichoa.row2box(row) for _, row in df.iterrows()
            ])
            return boxes

        _hoa_dets_cache = dict()
        annotataions = dict()

        for _, vid_info in vid_index.iterrows():
            seq_idx = vid_info.seq_idx
            annotation = dict()
            annotation['frame_idxs'] = [vid_info.start_frame]
            vid, _, nid = seq_idx
            if vid not in _hoa_dets_cache:
                hoa_dets = epichoa.load_video_hoa(
                    vid,
                    hoa_root=osp.join(self.epic_root, "hoa"))
                hoa_dets = hoa_dets[
                    hoa_dets.left < hoa_dets.right][
                        hoa_dets.top < hoa_dets.bottom]
                _hoa_dets_cache[vid] = hoa_dets
            else:
                hoa_dets = _hoa_dets_cache[vid]

            hoa_dets = hoa_dets[hoa_dets.frame == vid_info.start_frame]
            obj_df = hoa_dets[hoa_dets.det_type == 'object']
            if vid_info.side == 'left_hand':
                hand_df = hoa_dets.loc[(hoa_dets.det_type == 'hand') & (hoa_dets.side == 'left')]
            else:
                hand_df = hoa_dets.loc[(hoa_dets.det_type == 'hand') & (hoa_dets.side == 'right')]
            objects = _df2boxes(obj_df)
            hand = _df2boxes(hand_df)
            annotataions[seq_idx] = dict(
                bboxes_xyxy={
                    'objects': objects,
                    vid_info.side: hand,
                },
                frame_idxs=[vid_info.start_frame]
            )
        
        return vid_index, annotataions
            
    def _init_chunk_vid_index(self, vid_index):
        new_vid_index = vid_index.copy()
        new_vid_index['frame_idxs'] = [[0]] * len(vid_index)
        return new_vid_index

    def get_roi(self, video_annots, frame_ids, res=640, roi_expansion=1):
        """
        Get square ROI in xyxy format

        roi_expansion was 0.2
        """
        # Get all 2d points and extract bounding box with given image
        # ratio
        annots = self.annotations[video_annots.seq_idx]
        bboxes = [bboxs[frame_ids] for bboxs in annots["bboxes_xyxy"].values()]
        all_vid_points = np.concatenate(list(bboxes)) / self.resize_factor
        xy_points = np.concatenate(
            [all_vid_points[:, :2], all_vid_points[:, 2:]], 0)
        mins = xy_points.min(0)
        maxs = xy_points.max(0)
        roi_box_raw = np.array([mins[0], mins[1], maxs[0], maxs[1]])
        roi_bbox = bboxutils.bbox_wh_to_xy(
            bboxutils.make_bbox_square(bboxutils.bbox_xy_to_wh(roi_box_raw),
                                       bbox_expansion=roi_expansion))
        roi_center = (roi_bbox[:2] + roi_bbox[2:]) / 2
        # Assumes square bbox
        roi_scale = roi_bbox[2] - roi_bbox[0]
        affine_trans = handutils.get_affine_transform(roi_center, roi_scale,
                                                      [res, res])[0]
        return roi_bbox, affine_trans
    
    def get_interpolation_mask(self, vid, fid):
        path = f'{self._interpolation_dir}/{vid}/frame_{fid:010d}.png'
        mask = Image.open(path).convert('P')
        mask = mask.resize(self.image_size, Image.NEAREST)
        return mask
    
    def __getitem__(self, idx):
        return self.get_frame_info(idx)
    
    def get_frame_info(self, idx, res=640):
        vid_info = self.vid_index.iloc[idx]
        # Use all frames if frame_nb is -1
        if self.frame_nb == -1:
            frame_nb = vid_info.frame_nb
        else:
            frame_nb = self.frame_nb

        frame_ids = np.linspace(0, vid_info.frame_nb - 1,
                                frame_nb).astype(np.int)

        seq_frame_idxs = [self.annotations[vid_info.seq_idx]["frame_idxs"]][0]
        frame_idxs = [seq_frame_idxs[frame_id] for frame_id in frame_ids]
        video_id = vid_info.seq_idx[0]

        # Read images from tar file
        images = []
        masks = []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        roi, affine_trans = self.get_roi(vid_info, frame_ids)
        for frame_id in frame_ids:
            frame_idx = seq_frame_idxs[frame_id]

            img = self._read_frame(video_id, frame_idx)
            img = handutils.transform_img(img, affine_trans, [res, res])
            images.append(img)

            if self.with_interp_mask:
                mask = self.get_interpolation_mask(video_id, frame_idx)
                mask = handutils.transform_img(mask, affine_trans, [res, res])
                mask = np.asarray(mask)
                masks.append(mask)
            else:
                masks.append(None)

            obj_info, hand_infos, camera, setup = self.get_hand_obj_info(
                vid_info,
                frame_id,
                roi=roi,
                res=res,
                affine_trans=affine_trans)
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

        obs = dict(hands=collated_hand_infos,
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=setup,
                   frame_idxs=frame_idxs,
                   images=images,
                   masks=masks,
                   seq_idx=vid_info.seq_idx,
                   category=vid_info.object)

        return obs