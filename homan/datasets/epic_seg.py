from functools import partial, reduce
import json
from pathlib import Path
from glob import glob
import json

import numpy as np
from PIL import Image, ImageColor
import cv2


def read_epic_image(video_id, frame_idx, root='/home/skynet/Zhifan/data/epic_rgb_frames/'):
    root = Path(root)
    frame = root/video_id[:3]/video_id/f"frame_{frame_idx:010d}.jpg"
    frame = Image.open(frame)
    return np.asarray(frame)


class EpicSegGT(object):

    """
    For simplicity and for saving memory footprint, 
     we assume each frame has at most one instance per category,
    Thus, we store the SEG mask as a (H, W, C) ndarray,
     where mask[:, :, c] stands for the c-th class and is 1 if it's that object
    """

    """ json contains 307 names, only select those who occurs in `all_names`. """
    all_labels = [
        'left hand', 'right hand',
        'can',
        'cup'
        'plate',
        'bottle',
    ]

    cat2label = {label: i for i, label in enumerate(all_labels)}
    label2cat = {i: label for i, label in enumerate(all_labels)}
    
    def __init__(self,
                 seg_root,
                 hei=1080,
                 wid=1920):
        self.seg_root = Path(seg_root)
        self.image_hei = hei
        self.image_wid = wid
        self.num_classes = len(all_labels)
        self.path_map = dict()  # map video_id to json-path
        self.frame_collections = dict()

        for gt_file in glob(str(self.seg_root/'*.json')):
            name = gt_file.split('/')[-1]
            video_id = '_'.join(name.split('_')[1:3])
            self.path_map[video_id] = gt_file
            self.frame_collections[video_id] = set()

            with open(self.path_map[video_id]) as fp:
                data = json.load(fp)
            for frame_data in data:
                frame_name = frame_data['documents'][0]['name']  # length-1 documents
                frame_idx = int(frame_name.split('.')[0].split('_')[-1])
                self.frame_collections[video_id].add(frame_idx)
            
        self._cache = dict()    # 2 level map
                                # level 1: map from video_id to dict
                                # level 2: map from frame_idx to (H, W, C) ndarray
    
    def read_frame(self, video_id, frame_idx):
        """ 
        Args:
            video_id: str
            frame_idx: int
        
        Returns:
            (H, W, C) ndarray, where C = len(all_names)
            or 
            None if does not exists.
        """
        if video_id not in self.path_map:
            return None
        
        if video_id in self._cache:
            vid_info = self._cache[video_id]
            if frame_idx in vid_info:
                return vid_info[frame_idx]
        else:
            vid_info = dict()
        
        with open(self.path_map[video_id]) as fp:
            data = json.load(fp)

        kwargs = dict(hei=self.image_hei, wid=self.image_wid)
        for frame_data in data:
            frame_name = frame_data['documents'][0]['name']  # length-1 documents
            frame_ind = int(frame_name.split('.')[0].split('_')[-1])
            if frame_ind != frame_idx:
                continue
            annotation = frame_data['annotation']
            gt = Helper.walk_metas([annotation], **kwargs)[0]
            vid_info[frame_ind] = gt
        
        self._cache[video_id] = vid_info
        return self._cache[video_id][frame_idx]
    
    def avail_videos(self):
        return list(self.path_map.keys())
        
    def avail_frames(self, video_id):
        if video_id in self.path_map:
            return sorted(list(self.frame_collections[video_id]))
        else:
            return []


class EpicSegGTVisualizer(EpicSegGT):

    COLORMAP = {
        cat: clr
        for cat, clr in zip(
            EpicSegGT.all_labels, ImageColor.colormap.keys())
    }

    def __init__(self, *args, **kwargs):
        super(EpicSegGTVisualizer, self).__init__(*args, **kwargs)
    
    @property
    def colormap(self):
        """ mapping from category name to color. """
        return self.COLORMAP

    @staticmethod
    def color_to_array(color_hex):
        return tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    
    def get_segmentation(self, video_id, frame_idx, canvas=None):
        """

        Args:
            canvas: None or 

        Returns:
            segment_mask: ndarray of (H, W, 3) with COLORMAP
        """
        gt = self.read_frame(video_id, frame_idx)
        if canvas is None:
            canvas = np.zeros(
                [self.image_hei, self.image_wid, 3], dtype=np.uint8)
        for c in range(self.num_classes):
            color = self.colormap[self.label2cat[c]]
            code = ImageColor.colormap[color]
            color_arr = self.color_to_array(code[1:])  # remove '#'
            canvas[gt[..., c] == 1, :] = color_arr
        return canvas

    def get_image_overlay(self, video_id, frame_idx):
        image = read_epic_image(video_id, frame_idx)
        image = cv2.resize(image, (self.image_wid, self.image_hei))
        mask = self.get_segmentation(video_id, frame_idx, canvas=image)
        return cv2.addWeighted(image, 0.5, mask, 0.5, 1.0)
    

all_labels = EpicSegGT.all_labels
cat2label = EpicSegGT.cat2label


class Helper:

    """ Functions for processing annotation json. 

    walk_annotataions() 
        returns mask of (H, W) ndarray

    walk_blocks()
        returns mask by combining list of masks

    walk_entities(), walk_groups(), walk_metas()
        returns (H, W, C)

    """

    def walktree(l, func, **kwargs):
        """ Recursive walking throught a tree

        E.g. for walking annotationEntities:

        walk(annotationEntities, pop_name='func_annotationEntities', **kwargs)
        where kwargs is 
            {
                - 'func_annotationEntities': function
            }


        Args:
            l : list of dicts
            pop_name: str
        """
        res = []
        for d in l:
            res.append(func(d, **kwargs))
        return res
                
    def _func_annotations(d, **kwargs):
        return Helper.parse_segments(d['segments'], **kwargs)
    walk_annotations = partial(walktree, func=_func_annotations)

    def _func_blocks(d, **kwargs):
        mask_list = Helper.walk_annotations(d['annotations'], **kwargs)
        hei, wid = kwargs['hei'], kwargs['wid']
        return reduce(np.add, mask_list, np.zeros([hei, wid], np.int32))
    walk_blocks = partial(walktree, func=_func_blocks)

    def _func_entities(d, **kwargs):
        hei, wid = kwargs['hei'], kwargs['wid']
        gt = np.zeros([hei, wid, len(all_labels)], dtype=np.int32)
        name = d['name']  # TODO: change strict rule
        if name in cat2label:
            c = cat2label[name]
            mask_list = Helper.walk_blocks(d['annotationBlocks'], **kwargs)
            mask =  reduce(np.add, mask_list, np.zeros([hei, wid], np.int32))
            gt[..., c] = mask
        return gt
    walk_entities = partial(walktree, func=_func_entities)

    def _func_groups(d, **kwargs):
        hei, wid = kwargs['hei'], kwargs['wid']
        C = len(all_labels)
        mask_list = Helper.walk_entities(d['annotationEntities'], **kwargs)
        return reduce(np.add, mask_list, np.zeros([hei, wid, C], np.int32))
    walk_groups = partial(walktree, func=_func_groups)

    def _func_metas(d, **kwargs):
        hei, wid = kwargs['hei'], kwargs['wid']
        C = len(all_labels)
        mask_list = Helper.walk_groups(d['annotationGroups'], **kwargs)
        return reduce(np.add, mask_list, np.zeros([hei, wid, C], np.int32))
    walk_metas = partial(walktree, func=_func_metas)

    def parse_segments(segments, **kwargs):
        """ 
        Args:
            segments: list of 2d polygon
                where each 2d polygon is a list-of-list
            hei: int
            wid: int
        Returns:
            np.int32 array of (hei, wid)
        """
        hei, wid = kwargs['hei'], kwargs['wid']
        polygons = []
        for segment in segments:
            pts = np.asarray(segment, np.int32).reshape((-1, 1, 2))
            polygons.append(pts)
        mask = np.zeros([hei, wid], dtype=np.int32)
        mask = cv2.fillPoly(mask, polygons, (1,))
        return mask


if __name__ == '__main__':

    def get_all_names(json_root):

        walktree = Helper.walktree

        def func_entities(d, **kwargs):
            return d['name']
        walk_entities = partial(walktree, func=func_entities)

        def func_groups(d, **kwargs):
            name_list = walk_entities(d['annotationEntities'], **kwargs)
            name_set = set(n for n in name_list)
            return name_set
        walk_groups = partial(walktree, func=func_groups)

        def func_meta(d, **kwargs):
            name_set_list = walk_groups(d['annotationGroups'], **kwargs)
            return reduce(set.union, name_set_list, set())
        walk_meta = partial(walktree, func=func_meta)

        def func_json(d, **kwargs):
            meta_list = [d['annotation']]
            name_set_list = walk_meta(meta_list, **kwargs)
            return reduce(set.union, name_set_list, set())
        walk_json = partial(walktree, func=func_json)

        name_set = set()

        for jsonfile in glob(str(Path(json_root)/'*.json')):
            with open(jsonfile) as fp:
                data = json.load(fp)
            local_name_set = reduce(set.union, walk_json(data), set())
            name_set = name_set.union(local_name_set)
        return name_set
            
    # get_all_names()
    vis = EpicSegGTVisualizer('/home/skynet/Zhifan/data/Epic_zhifan/')
    seg = vis.get_segmentation('P12_04', 122)