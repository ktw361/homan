# -*- coding: utf-8 -*-

# pylint: disable=C0411,broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
import argparse
from collections import defaultdict
import logging
import tqdm
import os
import pickle

import cv2
import numpy as np
import torch

from libyana.exputils import argutils
from libyana.randomutils import setseeds

from homan import getdataset
from homan.eval import evalviz, pointmetrics, saveresults
from homan.jointopt import optimize_hand_object
from homan.lib2d import maskutils
from homan.pointrend import MaskExtractor
from homan.pose_optimization import find_optimal_poses
from homan.prepare.frameinfos import get_frame_infos, get_gt_infos
from homan.tracking import preprocess
from homan.utils.bbox import bbox_xy_to_wh, make_bbox_square
from homan.visualize import visualize_hand_object
from homan.viz import cliputils
from homan.viz.viz_gtpred_points import viz_gtpred_points
from handmocap.hand_mocap_api import HandMocap

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimize object meshes w.r.t. hand.")
    parser.add_argument("--dataset",
                        default="ho3d",
                        choices=[
                            "contactpose", "ho3d", "fhb", "epic", "core50",
                            "inhandycb", "ourycb"
                        ],
                        help="Dataset name")
    parser.add_argument("--epic_hdf5_root",
                        default=None)
    parser.add_argument("--epic_root", 
                        default="/home/skynet/Zhifan/datasets/epic")
    parser.add_argument("--epic_valid_path", 
                        default='/home/skynet/Zhifan/data/allVideos.xlsx')
    parser.add_argument("--dataset_mode",
                        default="chunk",
                        choices=["chunk", "vid"]
                        )
    parser.add_argument("--chunk_step",
                        default=4,
                        type=int,
                        help="Step between consecutive frames")
    parser.add_argument("--frame_nb",
                        default=10,
                        type=int,
                        help="Number of video frames to process in a batch, aka chunk_size")
    parser.add_argument("--data_step", default=100, type=int)
    parser.add_argument("--data_offset", default=0, type=int)
    parser.add_argument("--seq_idx", type=str,
                        help="manually select one video")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split",
                        default="val",
                        choices=["train", "val", "trainval", "test"],
                        help="Dataset name")
    parser.add_argument("--box_mode", choices=["gt", "track"], default="gt")
    parser.add_argument("--output_dir",
                        default="output",
                        help="Output directory.")
    parser.add_argument("--num_obj_iterations", default=50, type=int)
    parser.add_argument("--num_joint_iterations", default=201, type=int)
    parser.add_argument("--num_initializations", default=500, type=int)
    parser.add_argument("--mesh_path", type=str, help="Index of mesh ")
    parser.add_argument("--result_root", default="results/tmp")
    parser.add_argument(
        "--resume",
        help="Path to root folder of previously computed optimization results")
    parser.add_argument("--resume_indep", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viz_step", default=20, type=int)
    parser.add_argument("--save_indep", action="store_true")
    parser.add_argument("--only_missing", choices=[0, 1], type=int)
    parser.add_argument("--gt_masks", choices=[0, 1], default=0, type=int)
    parser.add_argument("--optimize_mano", choices=[0, 1], default=1, type=int)
    parser.add_argument("--optimize_mano_beta",
                        choices=[0, 1],
                        default=1,
                        type=int)
    parser.add_argument("--optimize_object_scale",
                        choices=[0, 1],
                        default=0,
                        type=int)
    parser.add_argument("--hand_proj_mode",
                        default="persp",
                        choices=["ortho", "persp"])
    parser.add_argument(
        "--lw_smooth",
        type=float,
        default=2000,
        help="Loss weight for smoothness.",
    )
    parser.add_argument(
        "--lw_v2d_hand",
        type=float,
        default=50,
        help="Loss weight for 2D vertices reprojection loss.",
    )
    parser.add_argument(
        "--lw_inter",
        type=float,
        default=1,
        help="Loss weight for coarse interaction loss.",
    )
    parser.add_argument(
        "--lw_contact",
        type=float,
        default=0,
        choices=[0, 1],
        help="Loss contact heuristic",
    )
    parser.add_argument(
        "--lw_depth",
        type=float,
        default=0,
        help="Loss weight for ordinal depth loss.",
    )
    parser.add_argument(
        "--lw_pca",
        type=float,
        default=0.004,
        help="Loss weight for PCA loss.",
    )
    parser.add_argument(
        "--lw_sil_obj",
        type=float,
        default=1,
        help="Loss weight for object mask loss.",
    )
    parser.add_argument(
        "--lw_sil_hand",
        type=float,
        default=0,
        help="Loss weight for hand mask loss.",
    )
    parser.add_argument(
        "--lw_collision",
        type=float,
        default=0,
        choices=[0, 0.001],
        help="Loss weight for collision loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_scale_obj",
        type=float,
        default=0.001,
        help="Loss weight for object scale loss. (None: default weight)",
    )
    parser.add_argument(
        "--lw_scale_hand",
        type=float,
        default=0.001,
        help="Loss weight for hand scale loss. (None: default weight)",
    )
    parser.add_argument("--hand_checkpoint",
                        default="extra_data/hand_module/pretrained_weights/"
                        "pose_shape_best.pth")
    parser.add_argument("--smpl_path", default="extra_data/smpl")
    args = parser.parse_args()
    args.lw_smooth_obj = args.lw_smooth
    args.lw_smooth_hand = args.lw_smooth
    argutils.print_args(args)
    if args.gt_masks and args.box_mode == "track":
        raise ValueError("gt_masks should not be used with bbox_mode 'track'")

    logger.info(f"Calling with args: {str(args)}")
    return args


class Fitter(object):
    def __init__(self,
                 image_size,
                 resume,
                 frame_nb=10,
                 debug=False,
                 gt_masks=0,
                 num_obj_iterations=50,
                 num_joint_iterations=201,
                 num_initializations=500,
                 hand_checkpoint="extra_data/hand_module/pretrained_weights/"
                 "pose_shape_best.pth",
                 smpl_path="extra_data/smpl",
                 ):
        """_summary_

        Args:
            resume (_type_): str Path or None
        """
        self.image_size = image_size
        self.resume = resume
        self.frame_nb = frame_nb
        self.debug = debug
        self.gt_masks = gt_masks
        self.num_obj_iterations = num_obj_iterations
        self.num_joint_iterations = num_joint_iterations
        self.num_initializations = num_initializations

        # Get pretrained networks
        self.mask_extractor = MaskExtractor(
                'detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl'
                )
        self.hand_predictor = HandMocap(hand_checkpoint, smpl_path)

    def _set_paths(self, sample_folder, result_root):
        self.sample_folder = sample_folder
        os.makedirs(sample_folder, exist_ok=True)
        self.save_path = os.path.join(result_root, "results.pkl")
        self.sample_path = os.path.join(sample_folder, "results.pkl")
        self.check_path = os.path.join(sample_folder, "joint_fit.pt")
    
    def fit_single_annot(self, annots, args, resume_folder):
        images_np, camintr, obj_bboxes, hand_bboxes, \
            gt_obj_verts, gt_hand_verts = self.preprocess_detections(annots)
        print("Regressing hands")

        camintr_nc = camintr.copy()
        camintr_nc[:, :2] = camintr_nc[:, :2] / self.image_size

        indep_fit_res, state_dict = self.get_indep_fit(
            resume_folder, images_np, camintr, obj_bboxes,
            hand_bboxes, annots)

        # Extract weight dictionary from arguments
        loss_weights = {
            key: val
            for key, val in vars(args).items() if "lw_" in key
        }

        # Run JOINT OPTIMIZATION
        # mano and object poses are initialized using indep_fit_res
        model, loss_evolution, imgs = optimize_hand_object(
            person_parameters=indep_fit_res["person_parameters"],
            object_parameters=indep_fit_res["object_parameters"],
            hand_proj_mode=args.hand_proj_mode,
            objvertices=indep_fit_res["obj_verts_can"],
            objfaces=indep_fit_res["obj_faces"],
            optimize_mano=args.optimize_mano,
            optimize_mano_beta=args.optimize_mano_beta,
            optimize_object_scale=args.optimize_object_scale,
            loss_weights=loss_weights,
            image_size=self.image_size,
            num_iterations=args.num_joint_iterations,
            images=images_np,
            camintr=camintr_nc,
            state_dict=state_dict,
            viz_step=args.viz_step,
            viz_folder=os.path.join(self.sample_folder, "jointoptim"),
        )
        save_dict = {
            "state_dict": {
                key: val.contiguous().cpu()
                for key, val in model.state_dict().items()
                if ("mano_model" not in key)
            }
        }
        torch.save(save_dict, os.path.join(
            self.sample_folder, "joint_fit.pt"))
        # Save initial optimization results
        # with open(indep_fit_path, "wb") as p_f:
        #     pickle.dump(indep_fit_res, p_f)

        init_obj_verts = model.verts_object_init
        init_hand_verts = model.verts_hand_init
        fit_obj_verts, _ = model.get_verts_object()
        fit_hand_verts, _ = model.get_verts_hand()

        gt_obj_verts = fit_obj_verts.new(gt_obj_verts)
        gt_hand_verts = fit_hand_verts.new(gt_hand_verts)
        self.visualize(model, images_np, gt_obj_verts, gt_hand_verts)
        self.save_results(
            args,
            init_obj_verts, init_hand_verts,
            fit_obj_verts, fit_hand_verts,
            gt_obj_verts, gt_hand_verts,
            model.faces_object, model.faces_hand,
            loss_evolution, imgs,
            super2d_img_path=indep_fit_res["super2d_img_path"])
            

    def preprocess_detections(self, annots):
        print("Pre-processing detections")
        images = annots["images"]
        right_hands = [
            hand for hand in annots["hands"] if hand["label"] == "right_hand"
        ]
        left_hands = [
            hand for hand in annots["hands"] if hand["label"] == "left_hand"
        ]
        setup = annots["setup"]

        camintr = annots["camera"]["K"].copy()
        # Get object bboxes and add padding
        obj_bboxes = np.array([annots["objects"][0]['bbox']])
        obj_bbox_padding = 5
        obj_bboxes = obj_bboxes + np.array([
            -obj_bbox_padding, -obj_bbox_padding, obj_bbox_padding,
            obj_bbox_padding
        ])
        gt_obj_verts = None
        if "objects" in setup:
            gt_obj_verts = np.concatenate(
                [annot["verts3d"] for annot in annots["objects"]], 1)
        gt_hand_verts = []
        if "right_hand" in setup:
            gt_hand_verts.append(
                np.concatenate([
                    hand["verts3d"] for hand in annots["hands"]
                    if hand["label"] == "right_hand"
                ], 1))
        if "left_hand" in setup:
            gt_hand_verts.append(
                np.concatenate([
                    hand["verts3d"]
                    for hand in annots["hands"] if hand["label"] == "left_hand"
                ], 1))

        # Get hand detections and make them square
        hand_bboxes = {}
        hand_expansion = 0.1
        if len(left_hands) > 0:
            hand_bboxes["left_hand"] = make_bbox_square(
                bbox_xy_to_wh(left_hands[0]['bbox']),
                bbox_expansion=hand_expansion)
        else:
            hand_bboxes["left_hand"] = None
        if len(right_hands) > 0:
            hand_bboxes["right_hand"] = make_bbox_square(
                bbox_xy_to_wh(right_hands[0]['bbox']),
                bbox_expansion=hand_expansion)
        else:
            hand_bboxes["right_hand"] = None

        # Preprocess images
        images_np = [
            preprocess.get_image(image, self.image_size) for image in images
        ]

        return images_np, camintr, obj_bboxes, hand_bboxes, \
            gt_obj_verts, gt_hand_verts

    def get_indep_fit(self,
                      resume_folder,
                      images_np,
                      camintr,
                      obj_bboxes,
                      hand_bboxes,
                      annots,
                      ):
        """_summary_

        Returns:
            indep_fit_res: dict of
                - 'person_parameters', list (len=frame_nb) of dict of
                    N = 1
                    - 'bboxes':		torch.Size([1, 4])
                    - 'cams':		torch.Size([1, 3])
                    - 'faces':		torch.Size([1, 1538, 3])
                    - 'local_cams':		torch.Size([1, 3])
                    - 'verts':		torch.Size([1, 778, 3])
                    - 'verts2d':		torch.Size([1, 778, 2])
                    - 'rotations':		torch.Size([1, 3, 3])
                    - 'mano_pose':		torch.Size([1, 45])
                    - 'mano_pca_pose':		torch.Size([1, 45])
                    - 'mano_rot':		torch.Size([1, 3])
                    - 'mano_betas':		torch.Size([1, 10])
                    - 'mano_trans':		torch.Size([1, 3])
                    - 'translations':		torch.Size([1, 1, 3])
                    - 'hand_side':          ['right']
                    - 'masks':		        torch.Size([1, 640, 640])
                    - 'K_roi':		        torch.Size([1, 3, 3])
                    - 'target_masks':		torch.Size([1, 256, 256])
                    - 'square_bboxes':		torch.Size([1, 4])

                - 'object_parameters': list (frame_nb) of dict of 
                    - 'rotations':		 torch.Size([1, 3, 3])
                    - 'translations':		 torch.Size([1, 1, 3])
                    - 'verts_trans':		 torch.Size([1, 2160, 3])
                    - 'target_masks':		 torch.Size([1, 256, 256])
                    - 'K_roi':		 torch.Size([1, 1, 3, 3])
                    - 'masks':		 torch.Size([1, 640, 640])
                    - 'verts':		 torch.Size([1, 2160, 3])
                    - 'full_mask':		 torch.Size([640, 640])
                
                - 'object_verts_can': ndarray (20, 2160, 3)
                
                - 'obj_faces': ndarray (20, 4120, 3)

                - 'super2d_img_path': str
                    e.g. 'results/epic/step1/samples/P08_21_0000000/detections_masks.png'

            state_dict: 
        """
        
        # Collect 2D and 3D evidence
        if resume_folder:
            # Load from previous computation
            resume_indep_path = os.path.join(resume_folder, "indep_fit.pkl")
            if self.resume:
                state_dict = None
            else:
                resume_joint_path = os.path.join(resume_folder, "joint_fit.pt")
                state_dict = torch.load(resume_joint_path)["state_dict"]
                state_dict = {
                    key: val.cuda()
                    for key, val in state_dict.items()
                }
            with open(resume_indep_path, "rb") as p_f:
                indep_fit_res = pickle.load(p_f)
            return indep_fit_res, state_dict

        person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(
            images_np,
            self.hand_predictor,
            self.mask_extractor,
            sample_folder=self.sample_folder,
            hand_bboxes=hand_bboxes,
            obj_bboxes=obj_bboxes,
            camintr=camintr,
            debug=self.debug,
            image_size=self.image_size,
        )

        super2d_img_path = os.path.join(self.sample_folder,
                                        "detections_masks.png")
        cv2.imwrite(super2d_img_path,
                    super2d_imgs[:, :, :3].astype(np.uint8)[:, :, ::-1])

        # For ablations, render ground truth object and hand masks
        if self.gt_masks:
            gt2d_imgs = get_gt_infos(images_np,
                                     annots,
                                     person_parameters=person_parameters,
                                     obj_mask_infos=obj_mask_infos,
                                     image_size=self.image_size,
                                     sample_folder=self.sample_folder)
            gt2d_img_path = os.path.join(self.sample_folder,
                                         "gt_detections_masks.png")

            cv2.imwrite(  # pylint: disable=E1101
                gt2d_img_path,
                gt2d_imgs[:, :, :3].astype(np.uint8)[:, :, ::-1])

        obj_verts_can = annots["objects"][0]['canverts3d']
        obj_faces = annots["objects"][0]['faces']

        # Compute object pose initializations w/ PHOSA's Diff Rendering
        object_parameters = find_optimal_poses(
            images=images_np,
            image_size=images_np[0].shape,
            vertices=obj_verts_can[0],
            faces=obj_faces[0],
            annotations=obj_mask_infos,
            num_initializations=self.num_initializations,
            num_iterations=self.num_obj_iterations,
            Ks=camintr,
            viz_path=os.path.join(self.sample_folder, "optimal_pose.png"),
            debug=self.debug,
        )

        # Populate person_parameters target_masks and K_roi given
        # object occlusions
        for person_param, obj_param, cam in zip(person_parameters,
                                                object_parameters,
                                                camintr):
            maskutils.add_target_hand_occlusions(
                person_param,
                obj_param,
                cam,
                debug=self.debug,
                sample_folder=self.sample_folder)

        indep_fit_res = {
            "person_parameters": person_parameters,
            "object_parameters": object_parameters,
            "obj_verts_can": obj_verts_can,
            "obj_faces": obj_faces,
            "super2d_img_path": super2d_img_path
        }
        # Save initial optimization results
        indep_fit_path = os.path.join(self.sample_folder, "indep_fit.pkl")
        with open(indep_fit_path, "wb") as p_f:
            pickle.dump(indep_fit_res, p_f)
        state_dict = None

        return indep_fit_res, state_dict
        
    def visualize(self,
                  model,
                  images_np,
                  gt_obj_verts,
                  gt_hand_verts,
                  ):
        frame_nb = self.frame_nb
        image_size = self.image_size
        
        viz_len = min(5, frame_nb)
        with torch.no_grad():
            frontal, top_down = visualize_hand_object(model,
                                                      images_np,
                                                      dist=4,
                                                      viz_len=frame_nb,
                                                      image_size=image_size)
            frontal_gt_only, top_down_gt_only = visualize_hand_object(
                model,
                images_np,
                dist=4,
                viz_len=frame_nb,
                image_size=image_size,
                verts_hand_gt=gt_hand_verts,
                verts_object_gt=gt_obj_verts,
                gt_only=True)

        with torch.no_grad():
            # GT + pred overlay renders
            frontal_gt, top_down_gt = visualize_hand_object(
                model,
                images_np,
                dist=4,
                verts_hand_gt=gt_hand_verts,
                verts_object_gt=gt_obj_verts,
                viz_len=frame_nb,
                image_size=image_size,
            )
            # GT + init overlay renders
            frontal_init_gt, top_down_init_gt = visualize_hand_object(
                model,
                images_np[:viz_len],
                dist=4,
                verts_hand_gt=gt_hand_verts,
                verts_object_gt=gt_obj_verts,
                viz_len=viz_len,
                init=True,
                image_size=image_size,
            )
        # pred verts need to be brought back from square image space
        viz_path = os.path.join(self.sample_folder, "final_points.png")
        viz_gtpred_points(images=images_np[:viz_len],
                          pred_images={
                              "frontal_pred": frontal[:viz_len],
                              "topdown_pred": top_down[:viz_len],
                              "frontal_pred+gt": frontal_gt[:viz_len],
                              "topdown_pred+gt": top_down_gt[:viz_len],
                              "frontal_init+gt": frontal_init_gt,
                              "topdown_init+gt": top_down_init_gt
                          },
                          save_path=viz_path)
        # Save predicted video clip
        top_down = cliputils.add_clip_text(top_down, "Pred")
        top_down_gt = cliputils.add_clip_text(top_down_gt, "Pred + GT")
        top_down_gt_only = cliputils.add_clip_text(top_down_gt_only,
                                                   "Ground Truth")

        clip = np.concatenate([
            np.concatenate([np.stack(images_np), frontal, frontal_gt_only], 2),
            np.concatenate([top_down_gt, top_down, top_down_gt_only], 2)
        ], 1)
        evalviz.make_video_np(clip,
                              viz_path.replace(".png", ".webm"),
                              resize_factor=0.5)
        optim_vid_path = os.path.join(self.sample_folder, "final_points.webm")
        evalviz.make_video_np(clip,
                              optim_vid_path.replace(".webm", ".mp4"),
                              resize_factor=0.5)
    
    def save_results(self,
                     args,
                     init_obj_verts,
                     init_hand_verts,
                     fit_obj_verts,
                     fit_hand_verts,
                     gt_obj_verts,
                     gt_hand_verts,
                     faces_object,
                     faces_hand,
                     loss_evolution,
                     imgs,
                     super2d_img_path,
                     ):

        with torch.no_grad():
            sample_obj_metrics = pointmetrics.get_point_metrics(
                gt_obj_verts, fit_obj_verts)
            sample_hand_metrics = pointmetrics.get_point_metrics(
                gt_hand_verts.view(-1, 778, 3),
                fit_hand_verts.view(-1, 778, 3))
            init_obj_metrics = pointmetrics.get_point_metrics(
                gt_obj_verts, init_obj_verts)
            init_hand_metrics = pointmetrics.get_point_metrics(
                gt_hand_verts.view(-1, 778, 3),
                init_hand_verts.view(-1, 778, 3))
            aligned_metrics = pointmetrics.get_align_metrics(
                gt_hand_verts.view(-1, 778, 3),
                fit_hand_verts.view(-1, 778, 3), gt_obj_verts, fit_obj_verts)
            init_aligned_metrics = pointmetrics.get_align_metrics(
                gt_hand_verts.view(-1, 778, 3),
                fit_hand_verts.view(-1, 778, 3), gt_obj_verts, fit_obj_verts)
            inter_metrics = pointmetrics.get_inter_metrics(
                fit_hand_verts, fit_obj_verts, faces_hand,
                faces_object)
            init_inter_metrics = pointmetrics.get_inter_metrics(
                init_hand_verts, init_obj_verts, faces_hand,
                faces_object)

        sample_metrics = {}
        # Metrics before joint optimization
        for key, vals in init_obj_metrics.items():
            sample_metrics[f"{key}_obj_init"] = vals
        for key, vals in init_hand_metrics.items():
            if key == "verts_dists":
                sample_metrics[f"{key}_hand_init"] = vals

        # Metrics after joint optimizations
        for key, vals in sample_obj_metrics.items():
            sample_metrics[f"{key}_obj"] = vals
        for key, vals in sample_hand_metrics.items():
            if key == "verts_dists":
                sample_metrics[f"{key}_hand"] = vals
        for key, vals in aligned_metrics.items():
            sample_metrics[key] = vals
        for key, vals in init_aligned_metrics.items():
            sample_metrics[f"{key}_init"] = vals
        for key, vals in inter_metrics.items():
            sample_metrics[f"{key}"] = vals
        for key, vals in init_inter_metrics.items():
            sample_metrics[f"{key}_init"] = vals

        all_metrics = defaultdict(list)
        for key, vals in sample_metrics.items():
            all_metrics[key].extend(vals)

        viz_path = os.path.join(self.sample_folder, "final_points.png")
        with open(self.sample_path, "wb") as p_f:
            pickle.dump(
                {
                    "opts": vars(args),
                    "losses": loss_evolution,
                    "metrics": sample_metrics,
                    "imgs": imgs,
                    "show_img_paths": {
                        "pred_gt": viz_path,
                        "super2d": super2d_img_path,
                        "last": imgs[max(list(imgs.keys()))]
                    },
                }, p_f)
        saveresults.dump(args, all_metrics, self.save_path)
        

def main(args):
    setseeds.set_all_seeds(args.seed)
    # Update defaults based on commandline args.
    dataset, image_size = getdataset.get_dataset(
        args.dataset,
        split=args.split,
        frame_nb=args.frame_nb,
        box_mode=args.box_mode,
        chunk_step=args.chunk_step,
        dataset_mode=args.dataset_mode,
        epic_hdf5_root=args.epic_hdf5_root,
        epic_root=args.epic_root,
        epic_valid_path=args.epic_valid_path,
    )
    print(f"Processing {len(dataset)} samples")

    fitter = Fitter(
        image_size=image_size,
        resume=args.resume,
        frame_nb=args.frame_nb,
        debug=args.debug,
        gt_masks=args.gt_masks,
        num_obj_iterations=args.num_obj_iterations,
        num_joint_iterations=args.num_joint_iterations,
        num_initializations=args.num_initializations,
        hand_checkpoint=args.hand_checkpoint,
        smpl_path=args.smpl_path,
    )

    for sample_idx in tqdm.trange(args.data_offset, len(dataset), args.data_step):
        # Prepare sample folder
        seq_idx = dataset.chunk_index.iloc[sample_idx]['seq_idx']
        annots = dataset[sample_idx]
        if dataset.name == 'epic':
            video_id = seq_idx[0]  # seq_idx = (video_id, _, _)
            start_frame = annots['frame_idxs'][0]
            sample_folder = os.path.join(args.result_root, "samples",
                                        f"{video_id}_{start_frame}")
        else:
            sample_folder = os.path.join(args.result_root, "samples",
                                        f"{seq_idx}_{sample_idx:08d}")
        if args.seq_idx:
            if seq_idx != args.seq_idx:
                continue
        fitter._set_paths(sample_folder, args.result_root)
        if args.only_missing and os.path.exists(fitter.check_path):
            print(f"Skipping existing {fitter.sample_path}")
            continue

        resume_folder = None
        if args.resume:
            if dataset.name == 'epic':
                resume_folder = os.path.join(args.resume, "samples",
                                            f"{video_id}_{start_frame}")
            else:
                resume_folder = os.path.join(args.resume, "samples",
                                            f"{seq_idx}_{sample_idx:08d}")

        fitter.fit_single_annot(annots, args, resume_folder)


if __name__ == "__main__":
    main(get_args())
