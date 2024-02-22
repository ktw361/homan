import os.path as osp
from functools import lru_cache
from typing import Tuple

import numpy as np
from libzhifan import io

CONTACT_STATES = {'no_contact',
                  'obj_to_obj_contact',
                  'object_contact',
                  'other_person_contact',
                  'self_contact'}
GRASPS = {'NP-Palm', 'NP-Fin', 'Pow-Pris', 'Pre-Pris',
          'Pow-Circ', 'Pre-Circ', 'Later', 'Other'}


def hos_mapping(raw_json: dict) -> dict:
    """ return: {frame[int]: preds[list]} """
    import re
    def get_int(s): return int(re.search('\d{10}', s).group(0))
    return {get_int(blob['file_name']): blob['predictions'] for blob in raw_json['images']}


class HOSGetter:

    def __init__(self):
        self.hos_root = '/media/barry/DATA/Zhifan/Sid-hand-objects/'

    def has_vid(self, vid):
        return osp.exists(osp.join(self.hos_root, f'{vid}.json'))

    @lru_cache(maxsize=4)
    def get_vid_hos(self, vid: str) -> dict:
        hos = hos_mapping(io.read_json(osp.join(self.hos_root, f'{vid}.json')))
        return hos

    def get_frame_hos_raw(self, vid, frame) -> dict:
        hos = self.get_vid_hos(vid)
        return hos[frame]

    def get_frame_hbox(self, vid, frame) -> Tuple[np.ndarray, np.ndarray]:
        """ Tight boxes

        hands: {LEFT/RIGHT_HAND: [x1, y1, x2, y2]}
        Output in 256x456 resolution
        """
        HAND_BBOX = 'hand_bbox'
        hos = self.get_vid_hos(vid)
        lbox, rbox = None, None
        for pred in hos[frame]:
            hand_side = pred['hand_side']
            hbox = np.asarray(pred[HAND_BBOX], dtype=np.float32)
            if hand_side == 'left_hand':
                lbox = hbox
            elif hand_side == 'right_hand':
                rbox = hbox
        return lbox, rbox

    def get_frame_detail(self, vid, frame):
        hos = self.get_vid_hos(vid)
        ldetail, rdetail = None, None
        for pred in hos[frame]:
            hand_side = pred['hand_side']
            if hand_side == 'left_hand':
                ldetail = pred
            elif hand_side == 'right_hand':
                rdetail = pred
        return ldetail, rdetail
