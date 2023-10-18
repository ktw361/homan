import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from libzhifan import io
from PIL import Image

# from lib.locators import ImageLocator, UnfilteredMaskLocator
from homan.datasets.epichor_reader_lib.locators import ImageLocator, UnfilteredMaskLocator

""" Interface

reader =  Reader(mask_version: str)
mask = reader.read_mask(vid, frame, keep_mask: None or List[str])
# mask is (H, W, N), where mapping is not applied
"""

_image_locator = ImageLocator()


class EpicImageReader:

    IMG_SIZE = (854, 480)

    def __init__(self, rgb_root='/media/skynet/DATA/Datasets/epic-100/rgb'):
        self.image_format = osp.join(
            rgb_root, '%s/%s/frame_%010d.jpg')
    
    def read_image(self, vid, frame) -> np.ndarray:
        img_pil = self.read_image_pil(vid, frame)
        if img_pil is None:
            return None
        return np.asarray(img_pil)

    def read_image_pil(self, vid, frame) -> Image:
        img_path = self.image_format % (vid[:3], vid, frame)
        if not osp.exists(img_path):
            return None
        return Image.open(img_path).resize(self.IMG_SIZE)


class Reader:
    """ Read VISOR Image and Mask  (NOT EPIC-KITCHENS)"""

    IMG_SIZE = (854, 480)

    def __init__(self,
                 mask_version: str,
                 data_root="/media/skynet/DATA/Datasets/visor-dense/"):
        """
        Args:
            mask_version: 'filtered' or 'unfiltered'
        """
        self.data_root = Path(data_root)
        self.image_root = self.data_root/"480p"
        if mask_version == 'filtered':
            self.mask_reader = FilteredMaskReader(data_root)
            self.mapping = self.mask_reader.mapping
        elif mask_version == 'unfiltered':
            self.mask_reader = UnfilteredMaskReader(data_root)
        else:
            raise ValueError(f"Unknown mask version: {mask_version}")

    def read_image(self, vid, frame) -> np.ndarray:
        img_pil = self.read_image_pil(vid, frame)
        if img_pil is None:
            return None
        return np.asarray(img_pil)

    def read_image_pil(self, vid, frame) -> Image.Image:
        fname = _image_locator.get_path(vid, frame)
        if fname is None:
            return None
        return Image.open(fname).resize(self.IMG_SIZE)

    def read_mask(self, vid, frame, return_mapping=False):
        """
        Returns:
            mask: (H, W, N) np.ndarray
            If return_mapping: 
                mapping: {category: int_id} where mask==int_id means category
        """
        return self.mask_reader.read_mask(
            vid, frame, return_mapping=return_mapping)

    def read_mask_pil(self, vid, frame) -> Image.Image:
        return self.mask_reader.read_mask_pil(vid, frame)

    def read_blend(self, vid, frame, alpha=0.5) -> Image.Image:
        """
        Returns: list of Image or Image
            (img, mask, overlay)
        """
        m = self.read_mask_pil(vid, frame)
        img_pil = self.read_image_pil(vid, frame)
        img = np.asarray(img_pil)
        m_vals = np.asarray(m)
        m_img_pil = m.convert('RGB')
        m_img = np.asarray(m_img_pil)
        m_img[m_vals == 0] = img[m_vals == 0]
        covered = Image.fromarray(m_img)
        blend = Image.blend(img_pil, covered, alpha)
        return blend


class FilteredMaskReader:

    def __init__(self,
                 data_root="/media/skynet/DATA/Datasets/visor-dense/"):
        self.data_root = Path(data_root)
        self.result_root = self.data_root/"interpolations"
        self.palette = Image.open(self.data_root/'meta_infos/00000.png').getpalette()
        self.mapping = io.read_json(self.data_root/"meta_infos/data_mapping.json")
    
    def read_mask(self, vid, frame, return_mapping=False):
        """
        Args:
            vid: e.g. P01_01
            frame: int
            return_mapping

        Returns:
            np.uint8 (H, W), where value ranges from {0, 1, ... N}
            [Optional] mapping
        """
        fname = f"{vid}_frame_{frame:010d}.png"
        fname = self.result_root/vid/fname
        if not osp.exists(fname):
            return None if not return_mapping else (None, None)
        mask = np.asarray(Image.open(fname)).astype(np.uint8)

        if return_mapping:
            mapping = self.mapping[vid]
            avail_ids = np.unique(mask)  # this include bg
            mapping = {k: v for k, v in mapping.items() if v in avail_ids}
            return mask, mapping

        return mask

    def read_mask_pil(self, vid, frame) -> Image.Image:
        m = self.read_mask(vid, frame)
        if m is None:
            return None
        m = Image.fromarray(m)
        m.putpalette(self.palette)
        return m


class UnfilteredMaskReader:

    def __init__(self,
                 data_root='/media/skynet/DATA/Datasets/visor-dense/'):
        self.data_root = Path(data_root)
        self.result_root = self.data_root/"unfiltered_interpolations"
        self.palette = Image.open(self.data_root/'meta_infos/00000.png').getpalette()
        self.unfiltered_locator = UnfilteredMaskLocator(
            result_root=self.result_root)
        self.mapping = pd.read_csv(self.data_root/"meta_infos/unfiltered_color_mappings.csv")

    def read_mask(self, vid, frame, return_mapping=False) -> np.ndarray:
        mask_path = self.unfiltered_locator.get_path(vid, frame)
        if mask_path is None:
            return None if not return_mapping else (None, None)
        if not osp.exists(mask_path):
            return None if not return_mapping else (None, None)
        mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
        if return_mapping:
            folder = self.unfiltered_locator.locate(vid, frame)
            df = self.mapping[self.mapping['interpolation'] == folder]
            mapping = {
                v['Object_name']: v['new_index']
                for i, v in df.iterrows()}
            return mask, mapping

        return mask

    def read_mask_pil(self, vid, frame) -> Image.Image:
        m = self.read_mask(vid, frame)
        if m is None:
            return None
        m = Image.fromarray(m)
        m.putpalette(self.palette)
        return m


def read_mask_with_keep(reader: Reader, 
                        vid, frame, 
                        keep: List[str],
                        return_pil=False):
    """ This might be expensive operation 

    Returns: (mask, mapping)
    """
    mask, mapping = reader.read_mask(vid, frame, return_mapping=True)
    if mask is None:
        return None, None
    avail_ids = np.unique(mask)
    keep_ids = [mapping[k] for k in keep if k in mapping]
    keep_mapping = {k: v for k, v in mapping.items() if v in keep_ids}

    for int_id in avail_ids:
        if int_id == 0:
            continue
        if int_id in keep_ids:
            mask[mask == int_id] = int_id
        else:
            mask[mask == int_id] = 0
    
    if return_pil:
        mask = Image.fromarray(mask)
        mask.putpalette(reader.mask_reader.palette)
    return mask, keep_mapping


if __name__ == '__main__':
    mask, mapping = read_mask_with_keep(Reader(mask_version='filtered'), 'P01_01', 28801, keep=['right hand', 'cup'], return_pil=True)
    print(mapping)
    mask