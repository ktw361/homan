import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data import transforms
from detectron2.structures import BitMasks, Instances, Boxes

from homan.pointrend import MaskExtractor
from homan.utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh, make_bbox_square
from homan.constants import REND_SIZE, BBOX_EXPANSION_FACTOR


""" 
Epic-kitchens interpolated mask reader.
"""


class InterpolatedMaskExtractor:
    
    def __init__(self):
        self.bbox_expansion = BBOX_EXPANSION_FACTOR

    def masks_from_bboxes(self,
                          masked_im,
                          boxes_wh,
                          pred_classes=None,
                          class_idx=-1,
                          input_format="RGB",
                          rend_size=REND_SIZE,
                          image_size=640):
        """
        EpicSegGT classes: [
            '_bg',
            'left hand', 
            'right hand',
            'can',
            'cup',
            'plate',
            'bottle',
            'mug',
        ]

        In original MaskExtractor: 'full_mask', 'pred_mask' and bit_masks have no difference

        Args:
            maked_im: Interpolated mask, (H, W)
                to be used as the `instance.pred_masks` in MaskExtractor
            class_idx: int, index of EpicSegGT

        Returns:
            list with 1 element of dict with
                - bbox: 
                    torch.float32 (4,)
                    Unchanged from input `boxes_wh`.
                - full_mask: 
                    torch.bool of (IMAGE_SIZE, IMAGE_SIZE), e.g. (640, 640)
                - square_box: 
                    np.float32 ndarray shape (4,)
                - crop_mask: 
                    np.bool ndarray shape (REND_SIZE, REND_SIZE), e.g. (256, 256)
                
                - class_id: UNUSED torch.int64, size ()
                - score: UNUSED torch.bool, size ()
        """
        num_inst = 1
        box = boxes_wh[0]
        bbox = bbox_wh_to_xy(box)
        square_bbox = make_bbox_square(bbox, self.bbox_expansion)
        square_boxes = torch.FloatTensor(
            np.tile(bbox_wh_to_xy(square_bbox),
                    (num_inst, 1)))

        masked_im[masked_im != class_idx] = 0
        masked_im[masked_im == class_idx] = 1
        masks = torch.from_numpy(masked_im)
        bit_masks = BitMasks(masks.unsqueeze(0))
        crop_masks = bit_masks.crop_and_resize(square_boxes,
                                               rend_size).clone().detach()
        bbox = bbox.astype(np.float32)
        square_bbox = square_bbox.astype(np.float32)
        return [dict(
            bbox=bbox,
            full_mask=masks,
            square_bbox=square_bbox,
            crop_mask=crop_masks[0].cpu().numpy(),
        )]


# class InterpolatedMaskExtractor(MaskExtractor):
    
#     def __init__(self):
#         # super(InterpolatedMaskExtractor, self).__init__
#         self.cfg = get_cfg()
#         self.aug = transforms.ResizeShortestEdge(
#             [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
#             self.cfg.INPUT.MAX_SIZE_TEST,
#         )

#     def masks_from_bboxes(self,
#                           masked_im,
#                           boxes_wh,
#                           pred_classes=None,
#                           class_idx=-1,
#                           input_format="RGB",
#                           rend_size=REND_SIZE,
#                           image_size=640):
#         """
#         Args:
#             class_idx (int): coco class index, -1 for using the most likely predicted class
#             boxes (np.array): (-1, 4) xyxy
#         Returns:
#             dict: {'square_boxes': (xywh)}
#         """
#         boxes_xy = [bbox_wh_to_xy(box) for box in boxes_wh]
#         model = self.predictor.model

#         # Initialize boxes
#         if not isinstance(boxes_xy, torch.Tensor):
#             boxes_xy = torch.Tensor(boxes_xy)
        
#         # assert pred_classes is None
#         if pred_classes is None:
#             pred_classes = class_idx * torch.ones(len(boxes_xy)).long()

#         # Clamp boxes to valid image region !
#         boxes_xy[:, :2].clamp_(0, max(masked_im.shape))
#         boxes_xy[:, 3].clamp_(0, masked_im.shape[0] + 1)
#         boxes_xy[:, 2].clamp_(0, masked_im.shape[1] + 1)
#         trans_boxes = Boxes(self.aug.get_transform(masked_im).apply_box(boxes_xy))
#         inp_im = self.preprocess_img(masked_im, input_format=input_format)
#         _, height, width = inp_im["image"].shape
#         instances = Instances(
#             image_size=(height, width),
#             pred_boxes=trans_boxes,
#             pred_classes=pred_classes,
#         )

#         # Preprocess image
#         inf_out = model.inference([inp_im], [instances])

#         # Extract masks
#         instance = inf_out[0]["instances"]
#         masks = instance.pred_masks
#         inst_boxes = instance.pred_boxes.tensor
#         try:
#             scores = instance.scores
#         except AttributeError:
#             scores = masks.new_ones(masks.shape[0])
#         pred_classes = instance.pred_classes
#         bit_masks = BitMasks(masks.cpu())
#         keep_annotations = []
#         full_boxes = torch.tensor([[0, 0, image_size, image_size]] *
#                                   len(inst_boxes)).float()
#         full_sized_masks = bit_masks.crop_and_resize(full_boxes, image_size)

#         for bbox_idx, box in enumerate(inst_boxes):
#             bbox = bbox_xy_to_wh(box.cpu())  # xy_wh
#             square_bbox = make_bbox_square(bbox, self.bbox_expansion)
#             square_boxes = torch.FloatTensor(
#                 np.tile(bbox_wh_to_xy(square_bbox),
#                         (len(instances), 1)))  # xy_xy
#             crop_masks = bit_masks.crop_and_resize(square_boxes,
#                                                    rend_size).clone().detach()
#             keep_annotations.append({
#                 "bbox":
#                 bbox,
#                 "class_id":
#                 pred_classes[bbox_idx],
#                 "full_mask":
#                 full_sized_masks[bbox_idx, :im.shape[0], :im.shape[1]].cpu(),
#                 "score":
#                 scores[bbox_idx],
#                 "square_bbox":
#                 square_bbox,  # xy_wh
#                 "crop_mask":
#                 crop_masks[0].cpu().numpy(),
#             })
#         return keep_annotations