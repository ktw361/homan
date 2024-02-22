""" Load hamer from disk """

import os

import pandas as pd
import numpy as np
import torch
from hydra.utils import to_absolute_path
from libzhifan.geometry import CameraManager, SimpleMesh, projection
from manopth.manolayer import ManoLayer
from pytorch3d.transforms import rotation_conversions as rot_cvt

from homan.datasets.epichor_lib.hos_getter import HOSGetter
from homan.datasets.epichor_reader_lib.reader import EpicImageReader
# from homan.homan_ManoModel import HomanManoModel
from homan.datasets.arctic_lib.manolayer_tracer import ManoLayerTracer
# from nnutils.handmocap import recover_pca_pose


# Src: hamer/datasets/utils.py
def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def compute_hand_translation(focal, pred_cam, box, out_w, out_h):
    """
    Args:
        focal: scalar
        pred_cam: [s, tx, ty]
        box: [xmin, ymin, xmax, ymax]
        out_w: scalar
        out_h: scalar
    """
    rescale_factor = 2.0
    center = (box[2:4] + box[0:2]) / 2.0
    scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
    center_x = center[0]
    center_y = center[1]
    BBOX_SHAPE = [192, 256]
    bbox_size = expand_to_aspect_ratio(
        scale*200, target_aspect_ratio=BBOX_SHAPE).max()

    # Making global translation
    s, tx, ty = pred_cam  # params['pred_cam']
    x0 = center_x - bbox_size/2
    y0 = center_y - bbox_size/2
    # putting 1/s and xmin is equivalent to putting center_x
    tx = tx + 1/s + (2*x0-out_w)/(s*bbox_size+1e-9)
    ty = ty + 1/s + (2*y0-out_h)/(s*bbox_size+1e-9)
    tz = 2*focal/(s*bbox_size+1e-9)
    global_transl = torch.Tensor([tx, ty, tz])
    return global_transl


class HamerLoader:
    """ Load hamer from disk """

    def __init__(self,
                 ho_version: str,
                 mano_root='extra_data/mano',
                 load_only=True,
                 v1_load_dir='/media/barry/DATA/Zhifan/epic_hor_data/hamer_hov1',
                 epichor_csv_path='/home/barry/Zhifan/epic_hor_method/code_epichor/image_sets/epichor_round3_2447valid.csv',
                 hoa_cache_path='/media/barry/DATA/Zhifan/epic_hor_data/cache/hoa_hbox.pth',
                 ):
        """
        Args:
            ho_version: 'v1' or 'v2'
            load_dir: e.g. <load_dir>/P01_01/frame_0000012345.pt
        """
        self.ho_version = ho_version
        if ho_version == 'v1':
            self.load_dir = v1_load_dir
            df = pd.read_csv(epichor_csv_path)
            hoa_cache = torch.load(hoa_cache_path)  # xywh in 1920x1080

            def to_hierarchy(df, hoa_cache):
                all_boxes = dict()
                for i, row in df.iterrows():
                    mp4_name = row['mp4_name']
                    vid = row['vid']
                    side = row['handside'].replace(' hand', '')
                    if vid not in all_boxes:
                        all_boxes[vid] = dict()
                    if side not in all_boxes[vid]:
                        all_boxes[vid][side] = dict()
                    for frame, box in hoa_cache[mp4_name].items():
                        all_boxes[vid][side][frame] = box
                return all_boxes
            
            self.all_boxes = to_hierarchy(df, hoa_cache)
            self.box_resolution = np.float32([1920, 1080] * 2)

        elif ho_version == 'v2':
            self.load_dir = to_absolute_path('./data/hamer_hov2')
            self.hosgetter = HOSGetter()
            self.box_resolution = np.float32([456, 256] * 2)

        if not load_only:  # for debugging
            self.reader = EpicImageReader()
            self.left_mano = ManoLayer(
                flat_hand_mean=True, ncomps=45, side='left',
                mano_root=mano_root, use_pca=False)
            self.right_mano = ManoLayer(
                flat_hand_mean=True, ncomps=45, side='right',
                mano_root=mano_root, use_pca=False)
        
        self.l_mano_tracer = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side='left', use_pca=True,
            mano_root=mano_root).cuda()
        self.r_mano_tracer = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side='right', use_pca=True,
            mano_root=mano_root).cuda()
        
        # Used for recovering pca
        # self.left_pca_aux = ManoLayer(
        #     flat_hand_mean=False, ncomps=45, side='left',
        #     mano_root=mano_root, use_pca=False)
        # self.right_pca_aux = ManoLayer(
        #     flat_hand_mean=False, ncomps=45, side='right',
        #     mano_root=mano_root, use_pca=False)

    def load_frame_all_params(self, vid: str, frame: int) -> dict:
        """ Load both hand params from *.pt file """
        return torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')

    def avail_frames(self, vid: str) -> list:
        """ Return available frames for a video """
        return sorted([
            int(f.split('.')[0].split('_')[-1])
            for f in os.listdir(f'{self.load_dir}/{vid}')])

    def has_frame(self, vid: str, frame: int) -> bool:
        """ Check if a frame is available """
        return os.path.exists(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')

    def _load_hamer_box(self, vid, frame, is_left) -> np.ndarray:
        """ return xyxy in 1920x1080 / 256x456 resolution """
        if self.ho_version == 'v1':
            x1, y1, w, h = self.all_boxes[vid]['left' if is_left else 'right'][frame]
            box = np.array([x1, y1, x1+w, y1+h])
            return box
        elif self.ho_version == 'v2':
            lbox, rbox = self.hosgetter.get_frame_hbox(vid, frame)
            return lbox if is_left else rbox

    def _load_hamer_params(self, vid, frame, is_left) -> dict:
        """ src: <hamer-repo>/infer_epic/infer.py"""
        if not self.has_frame(vid, frame):
            return None
        all_params = torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
        hand_side = 'left' if is_left else 'right'
        if hand_side not in all_params:
            return None
        return all_params[hand_side]

    def get_hamer_parmas(self, global_cam, vid, frame_inds: list, is_left: bool,
                         device='cuda'):
        """  This will return params to be used by
        HomanManoModel(mean = False, pca = True)

        Originally HAMER renders with mean=True and pca=False, we do the convertion

        Returns:
            mano_pca_pose: (N, 45)
            mano_hand_betas: (N, 10)
            hand_rotation_6d: (N, 6) apply-to-col
            hand_translation: (N, 1, 3)
        """
        out_w = global_cam.img_w
        out_h = global_cam.img_h
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / self.box_resolution

        hand_poses = [None for _ in frame_inds]
        mano_betas = [None for _ in frame_inds]
        global_orients = [None for _ in frame_inds]
        global_translations = [None for _ in frame_inds]

        for i, frame in enumerate(frame_inds):
            box = self._load_hamer_box(vid, frame, is_left)
            box = box * box_scale
            params = self._load_hamer_params(vid, frame, is_left)
            global_orients[i] = rot_cvt.matrix_to_axis_angle(params['global_orient'])
            hand_poses[i] = rot_cvt.matrix_to_axis_angle(params['hand_pose']).view(1, 45)
            global_translations[i] = compute_hand_translation(
                global_cam.fx, params['pred_cam'], box, out_w, out_h)
            mano_betas[i] = params['betas'].view(1, 10)

        global_orients = torch.cat(global_orients, axis=0)  # (N, 3)
        hand_poses = torch.cat(hand_poses, axis=0)  # (N, 45)
        global_translations = torch.cat(global_translations, axis=0)  # (N, 3)
        mano_betas = mano_betas[0].repeat(len(frame_inds), 1)  # Shape follows first frame
        if is_left:
            global_orients[:, 1:] *= -1
            hand_poses = hand_poses.view(-1, 15, 3) * torch.Tensor([1, -1, -1])
            hand_poses = hand_poses.view(-1, 45)
        hand_poses = hand_poses.to(device)
        mano_betas = mano_betas.to(device)
        global_orients = global_orients.to(device)
        global_translations = global_translations.to(device)
        thetas = torch.cat([global_orients, hand_poses], axis=1)

        mano_tracer = self.l_mano_tracer if is_left else self.r_mano_tracer
        _, _, T_world = mano_tracer.forward_transform(thetas, root_palm=True)
        T_rot = T_world[:, :3, :3]
        T_transl = T_world[:, :3, -1]
        hand_translation = T_transl.view(-1, 1, 3) + global_translations.view(-1, 1, 3)
        hand_rotation_6d = rot_cvt.matrix_to_rotation_6d(T_rot)

        # mano_pca_pose = self.recover_pca_pose(hand_poses, is_left)
        M_pca_inv = torch.inverse(mano_tracer.th_comps)
        mano_pca_pose = (hand_poses - mano_tracer.th_hands_mean).mm(M_pca_inv)

        return mano_pca_pose, hand_poses, mano_betas, hand_rotation_6d, hand_translation
    
    # def recover_pca_pose(self, pred_hand_pose: torch.Tensor, is_left: bool) -> torch.Tensor:
    #     """
    #     if
    #         v_exp = ManopthWrapper(pca=False, flat=False).(x_0)
    #         x_pca = self.recover_pca_pose(self.x_0)  # R^45
    #     then
    #         v_act = ManoLayer(pca=True, flat=False, ncomps=45).forward(x_pca)
    #         v_exp == v_act

    #     note above requires mano_rot == zeros, since the computation of rotation
    #         is different in ManopthWrapper
    #     """
    #     mano_layer_aux = self.left_pca_aux if is_left else self.right_pca_aux
    #     M_pca_inv = torch.inverse(mano_layer_aux.th_comps)
    #     mano_pca_pose = pred_hand_pose.mm(M_pca_inv)
    #     return mano_pca_pose

    def visualise_frame(self,
                        vid: str, frame: int,
                        hand_side: str,
                        out_h=256, out_w=456,
                        ret_mesh=False) -> np.ndarray:
        """
        Assumes each *.pt contains:
        dict(left=params, right=params)
        params = dict(
            pred_cam
                Tensor torch.Size([3])
            global_orient
                Tensor torch.Size([1, 3, 3])
            hand_pose
                Tensor torch.Size([15, 3, 3])
            betas
                Tensor torch.Size([10])
        )

        Args:
            hand_side: 'left' or 'right'
        """
        box = self._load_hamer_box(vid, frame, 'left' in hand_side)
        params = self._load_hamer_params(vid, frame, 'left' in hand_side)
        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])

        glb_orient = rot_cvt.matrix_to_axis_angle(params['global_orient'])
        thetas = rot_cvt.matrix_to_axis_angle(
            params['hand_pose']).view([1, 45])
        thetas = torch.cat([glb_orient, thetas], axis=1)
        betas = params['betas'].view([1, 10])

        if hand_side == 'left':
            mano = self.left_mano
            # The following is to flip the hand (flipping y and z)
            thetas = thetas.view(16, 3)
            thetas[:, 1:] *= -1
            thetas = thetas.view([1, 48])
        elif hand_side == 'right':
            mano = self.right_mano
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / self.box_resolution
        box *= box_scale

        vh, jh = mano.forward(thetas, betas)
        vh /= 1000.
        jh /= 1000.

        epic_focal = 5000  # Please make this infinite
        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)
        global_transl = compute_hand_translation(
            epic_focal, params['pred_cam'], box, out_w, out_h)

        mesh = SimpleMesh(vh + global_transl, mano.th_faces)
        if ret_mesh:
            return mesh
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend

    def vis_debug_homan(self,
                        vid: str, frame: int,
                        hand_side: str,
                        out_h=256, out_w=456,
                        ret_mesh=False) -> np.ndarray:
        """ This implement the forward pass with HomanManoModel(mean=False, pca=True)
        note: visualise_frame() uses ManoLayer(mean=True, pca=False)

        Args:
            hand_side: 'left' or 'right'
        """
        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])
        epic_focal = 5000  # Please make this infinite
        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)
        
        mano_tracer = self.l_mano_tracer if hand_side == 'left' else self.r_mano_tracer
        mano_pca_pose, mano_betas, hand_rotation_6d, hand_translation = \
            self.get_hamer_parmas(global_cam, vid, [frame], hand_side == 'left')
        th_pose_coeffs = torch.cat(
            [mano_pca_pose.new_zeros([1, 3]), mano_pca_pose], -1)
        vh, _, _center_jts = mano_tracer.forward(th_pose_coeffs, mano_betas)  # (1, 778, 3)
        vh = vh / 1000.

        rot_mat = rot_cvt.rotation_6d_to_matrix(hand_rotation_6d)
        vh = vh @ rot_mat.permute(0, 2, 1) + hand_translation

        mesh = SimpleMesh(vh[0], mano_tracer.th_faces)
        if ret_mesh:
            return mesh
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend

        glb_orient = rot_cvt.matrix_to_axis_angle(params['global_orient'])
        thetas = rot_cvt.matrix_to_axis_angle(
            params['hand_pose']).view([1, 45])
        thetas = torch.cat([glb_orient, thetas], axis=1)
        betas = params['betas'].view([1, 10])

        if hand_side == 'left':
            # mano = self.left_mano
            box = lbox
            # The following is to flip the hand (flipping y and z)
            thetas = thetas.view(16, 3)
            thetas[:, 1:] *= -1
            thetas = thetas.view([1, 48])
            # Note: flat mean is False
        elif hand_side == 'right':
            # mano = self.right_mano
            box = rbox

        mano_tracer = ManoLayerTracer(
            flat_hand_mean=True, ncomps=45, side=hand_side, use_pca=False,
            mano_root=to_absolute_path('./externals/mano/'))
        
        # homan mano specific
        mano_pose = thetas[:, 3:]  # (1, 45)
        # _mano_layer = homan_mano.mano_layer
        # M_pca_inv = torch.inverse(mano_tracer.th_comps)
        # mano_pca_pose = (mano_pose - mano_tracer.th_hands_mean).mm(M_pca_inv)
        th_pose_coeffs = torch.cat(
            [mano_pose.new_zeros([1, 3]), mano_pose], -1)
        vh, j = mano_tracer.forward(th_pose_coeffs, betas)
        vh = vh / 1000.
        # vh = mano_res['verts']

        _, _, T_world = mano_tracer.forward_transform(thetas, root_palm=True)
        T_rot = T_world[:, :3, :3]
        T_transl = T_world[:, :3, 3:]
        vh = (torch.bmm(T_rot, vh.permute(0, 2, 1)) + T_transl).permute(0, 2, 1)

        # global_rotation = rot_cvt.axis_angle_to_matrix(glb_rot)

        box_scale = np.float32([out_w, out_h, out_w, out_h]) / \
            np.float32([456, 256, 456, 256])
        box *= box_scale
        global_transl = compute_hand_translation(
            epic_focal, params['pred_cam'], box, out_w, out_h)

        vertices = vh + global_transl
        faces = mano_tracer.th_faces

        mesh = SimpleMesh(vertices, faces)
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend