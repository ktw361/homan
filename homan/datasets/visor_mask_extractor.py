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


class VisorMaskExtractor:
    
    def __init__(self):
        self.bbox_expansion = BBOX_EXPANSION_FACTOR
        """ These two will be set as the input stage, they act for passing data and to not destroy interface 

            maked_im: Interpolated mask, (H, W)
                to be used as the `instance.pred_masks` in MaskExtractor
            class_idx: int, index of EpicSegGT
        """
        self._idx = None
        self._mask_hand = None  # a list of np.ndarray
        self._mask_obj = None

    def masks_from_bboxes(self,
                          im,
                          boxes_wh,
                          pred_classes=None,
                          class_idx=-1,
                          input_format="RGB",
                          rend_size=REND_SIZE,
                          image_size=640):
        """
        In original MaskExtractor: 'full_mask', 'pred_mask' and bit_masks have no difference

        Args:
            class_idx: 0 for hand, -1 for obj

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
        bbox = boxes_wh[0]
        square_bbox = make_bbox_square(bbox, self.bbox_expansion)
        square_boxes = torch.FloatTensor(
            np.tile(bbox_wh_to_xy(square_bbox),
                    (num_inst, 1)))

        if class_idx == -1:
            masks = torch.from_numpy(self._mask_obj[self._idx]).clone()
        elif class_idx == 0:
            masks = torch.from_numpy(self._mask_hand[self._idx]).clone()
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