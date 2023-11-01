# -*- coding: utf-8 -*-

# pylint: disable=C0411,broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
import argparse
from collections import defaultdict
import logging
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points

from libyana.exputils import argutils
from libyana.randomutils import setseeds

from homan import getdataset
from homan.eval import evalviz, pointmetrics, saveresults
from homan.jointopt import optimize_hand_object
from homan.lib2d import maskutils
from homan.pointrend import MaskExtractor
from homan.arctic_pose_optimization import find_optimal_poses
from homan.prepare.frameinfos import get_frame_infos, get_gt_infos
from homan.tracking import preprocess
from homan.utils.bbox import bbox_xy_to_wh, make_bbox_square
from homan.visualize import visualize_hand_object
from homan.viz import cliputils
from homan.viz.viz_gtpred_points import viz_gtpred_points
from homan.datasets.visor_mask_extractor import VisorMaskExtractor
from handmocap.hand_mocap_api import HandMocap
import pandas as pd

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimize object meshes w.r.t. hand.")
    parser.add_argument("--dataset",
                        default="arctic_stable",
                        choices=[
                            "arctic_stable"
                        ],
                        help="Dataset name")
    parser.add_argument("--frame_nb",
                        default=30,
                        type=int,
                        help="Number of video frames to process in a batch")
    parser.add_argument("--learnt_obj_pose", choices=[0, 1], default=1)
    parser.add_argument("--gt_mano", choices=[0, 1], default=1, type=int)

    parser.add_argument("--box_mode", choices=["gt", "track"], default="gt")
    parser.add_argument("--gt_masks", choices=[0, 1], default=1, type=int)
    parser.add_argument("--data_step", default=1, type=int)
    parser.add_argument("--data_offset", default=0, type=int)
    parser.add_argument("--data_stop", default=99999, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--split",
                        default="test",
                        choices=["train", "val", "trainval", "test"],
                        help="Dataset name")
    parser.add_argument("--output_dir",
                        default="output",
                        help="Output directory.")
    parser.add_argument("--num_obj_iterations", default=0, type=int)
    parser.add_argument("--num_joint_iterations", default=0, type=int)
    parser.add_argument("--num_initializations", default=50, type=int)
    parser.add_argument("--mesh_path", type=str, help="Index of mesh ")
    parser.add_argument("--result_root", default="results/arctic_stable")
    parser.add_argument(
        "--resume",
        help="Path to root folder of previously computed optimization results")
    parser.add_argument("--resume_indep", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viz_step", default=20, type=int)
    # parser.add_argument("--save_indep", action="store_true")
    parser.add_argument("--only_missing", choices=[0, 1], type=int)

    parser.add_argument("--optimize_mano", choices=[0, 1], default=0, type=int)
    parser.add_argument("--optimize_mano_beta",
                        choices=[0, 1],
                        default=0,
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
        default=0, # default=50,
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


def main(args):
    setseeds.set_all_seeds(args.seed)
    # Update defaults based on commandline args.
    dataset, image_size = getdataset.get_dataset(
        args.dataset,
        split=None,
        frame_nb=args.frame_nb,
        box_mode=args.box_mode,
        chunk_step=None,
        epic_mode=None,
    )
    print(f"Processing {len(dataset)} samples")
    # Get pretrained networks
    # mask_extractor = MaskExtractor()
    mask_extractor = VisorMaskExtractor()
    hand_predictor = HandMocap(args.hand_checkpoint, args.smpl_path)

    all_metrics = defaultdict(list)
    data_stop = min(len(dataset), args.data_stop)
    for sample_idx in range(args.data_offset, data_stop, args.data_step):
        annots = dataset[sample_idx]
        vid_start_end = annots['annot_full_key']

        sample_folder = os.path.join(args.result_root, "samples", vid_start_end)
        os.makedirs(sample_folder, exist_ok=True)
        save_path = os.path.join(args.result_root, "results.pkl")
        sample_path = os.path.join(sample_folder, "results.pkl")
        check_path = os.path.join(sample_folder, "epichor_metric.csv")
        if args.only_missing and os.path.exists(check_path):
            print(f"Skipping existing {sample_path}")
            continue

        print("Pre-processing detections")
        images = annots["images"]
        right_hands = [
            hand for hand in annots["hands"] if hand["label"] == "right_hand"
        ]
        left_hands = [
            hand for hand in annots["hands"] if hand["label"] == "left_hand"
        ]
        setup = annots["setup"]
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

        camintr = annots["camera"]["K"].copy()
        # Get object bboxes and add padding
        obj_bboxes = np.array([annots["objects"][0]['bbox']])
        obj_bbox_padding = 5
        obj_bboxes = obj_bboxes + np.array([
            -obj_bbox_padding, -obj_bbox_padding, obj_bbox_padding,
            obj_bbox_padding
        ])

        # Preprocess images
        images_np = [
            preprocess.get_image(image, image_size) for image in images
        ]
        print("Regressing hands")

        camintr_nc = camintr.copy()
        camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

        # indep_fit_path = os.path.join(sample_folder, "indep_fit.pkl")
        # Collect 2D and 3D evidence
        if not args.resume:
            mask_extractor._mask_hand = annots['masks_hand']
            mask_extractor._mask_obj = annots['masks_obj']

            det_person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(
                images_np,
                None, # hand_predictor,
                mask_extractor,
                sample_folder=sample_folder,
                hand_bboxes=hand_bboxes,
                obj_bboxes=obj_bboxes,
                camintr=camintr,
                debug=args.debug,
                image_size=image_size,
            )
            if args.gt_mano:
                person_parameters = annots['gt_person_parameters']
                for i in range(len(person_parameters)):
                    for k, v in person_parameters[i].items():
                        if isinstance(v, torch.Tensor):
                            person_parameters[i][k] = v.cuda()
                for i in range(len(det_person_parameters)):
                    person_parameters[i]['masks'] = det_person_parameters[i]['masks']
                    person_parameters[i]['cams'] = torch.ones([1, 3], device='cuda', dtype=torch.float32)  # not really used
                    # person_parameters[i]['cams'] = det_person_parameters[i]['cams']  # not really used
            else:
                person_parameters = det_person_parameters

            super2d_img_path = os.path.join(sample_folder,
                                            "detections_masks.png")
            cv2.imwrite(super2d_img_path,
                        super2d_imgs[:, :, :3].astype(np.uint8)[:, :, ::-1])

            # For ablations, render ground truth object and hand masks
            if args.gt_masks:
                gt2d_imgs = get_gt_infos(images_np,
                                         annots,
                                         person_parameters=person_parameters,
                                         obj_mask_infos=obj_mask_infos,
                                         image_size=image_size,
                                         sample_folder=sample_folder)
                gt2d_img_path = os.path.join(sample_folder,
                                             "gt_detections_masks.png")

                cv2.imwrite(  # pylint: disable=E1101
                    gt2d_img_path,
                    gt2d_imgs[:, :, :3].astype(np.uint8)[:, :, ::-1])

            obj_verts_can = annots["objects"][0]['canverts3d']
            obj_faces = annots["objects"][0]['faces']

            rotations_inits = annots['rotations_inits'].cuda()
            translations_inits = annots['translations_inits'].cuda()

            # Compute object pose initializations
            object_parameters = find_optimal_poses(
                images=images_np,
                image_size=images_np[0].shape,
                vertices=obj_verts_can[0],
                faces=obj_faces[0],
                annotations=obj_mask_infos,
                rotations_inits=rotations_inits,
                translations_inits=translations_inits,
                # num_initializations=args.num_initializations,
                num_iterations=args.num_obj_iterations,
                Ks=camintr,
                viz_path=os.path.join(sample_folder, "optimal_pose.png"),
                debug=args.debug,
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
                    debug=args.debug,
                    sample_folder=sample_folder)

            indep_fit_res = {
                "person_parameters": person_parameters,
                "object_parameters": object_parameters,
                "obj_verts_can": obj_verts_can,
                "obj_faces": obj_faces,
                "super2d_img_path": super2d_img_path
            }
            # Save initial optimization results
            # with open(indep_fit_path, "wb") as p_f:
            #     pickle.dump(indep_fit_res, p_f)
            state_dict = None

        # Extract weight dictionary from arguments
        loss_weights = {
            key: val
            for key, val in vars(args).items() if "lw_" in key
        }

        # Run step-1 joint optimization
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
            image_size=image_size,
            num_iterations=args.num_joint_iterations,
            images=images_np,
            camintr=camintr_nc,
            state_dict=state_dict,
            viz_step=args.viz_step,
            viz_folder=None, #os.path.join(sample_folder, "jointoptim"),
            optimize_hand_pose=False,
        )

        # Load the results
        homan_object_poses = torch.load(os.path.join(sample_folder, "homan_object_poses.pth"))
        model.translations_object = nn.Parameter(homan_object_poses['translations_object'])
        model.rotations_object = nn.Parameter(homan_object_poses['rotations_object'])
        model.cuda()

        # EPIC-HOR metrics
        with torch.no_grad():
            verts_pred, _ = model.get_verts_object()  # (N, V, 3)
            verts_gt = annots['objects'][0]['verts3d'].cuda()
            diameter = annots['diameter']
            # ADD & ADD-CLS
            v2v_dist = ((verts_gt - verts_pred)**2).sum(dim=2).sqrt()  # (N, V)
            add_010 = (v2v_dist < 0.10 * diameter).sum(dim=1).float() / v2v_dist.size(1)  # (N,)
            add_010_mean = add_010.mean()
            add_cls_010 = (v2v_dist.mean() < 0.10 * diameter).float()  # (N,)
            # ADD-S & ADD-S-CLS
            v2nn_dist = knn_points(verts_pred, verts_gt, K=1).dists
            add_s_010 = (v2nn_dist < 0.10 * diameter).sum(dim=1).float() / v2nn_dist.size(1)
            add_s_010_mean = add_s_010.mean()
            add_s_cls_010 = (v2nn_dist.mean() < 0.10 * diameter).float()

            add_s_002 = (v2nn_dist < 0.02 * diameter).sum(dim=1).float() / v2nn_dist.size(1)
            add_s_002_mean = add_s_002.mean()
            add_s_cls_002 = (v2nn_dist.mean() < 0.02 * diameter).float()

            # IOU
            _, sil_metric_dict = model.losses.compute_sil_loss_object(
                verts=verts_pred, faces=model.faces_object)
            iou = sil_metric_dict['iou_object']
            _, sil_metric_dict = model.losses.compute_sil_loss_object(
                verts=verts_pred, faces=model.faces_object)
            iou = sil_metric_dict['iou_object']
        rows = [
            dict(
                iou=iou, 
                add_010=add_010_mean.item(),
                add_cls_010=add_cls_010.item(),
                add_s_010=add_s_010_mean.item(),
                add_s_cls_010=add_s_cls_010.item(),
                add_s_002=add_s_002_mean.item(),
                add_s_cls_002=add_s_cls_002.item(),
            )
        ]
        df = pd.DataFrame(rows)
        print(df)
        # pd.DataFrame(rows).to_csv(
        #     os.path.join(sample_folder, "epichor_metric.csv"))
        # Save lightweight poses
        # homan_object_poses = {
        #     'translations_object': model.translations_object.cpu(),
        #     'rotations_object': model.translations_object.cpu(),
        # }
        # torch.save(homan_object_poses, os.path.join(sample_folder, "homan_object_poses.pth"))


if __name__ == "__main__":
    main(get_args())
