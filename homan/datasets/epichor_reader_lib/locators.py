import bisect
import os
import os.path as osp
import pickle
import re
from pathlib import Path
from typing import Union

import numpy as np
import tqdm
from hydra.utils import to_absolute_path


class PairLocator:
    """ locate a (vid, frame) in P01_01_0003
    See also ImageLocator and UnfilteredMaskLocator

    Interface:
    locator = PairLocator(
        result_root='/home/skynet/Zhifan/data/visor-dense/480p',
        cache_path='.cache/image_pair_index.pkl')
    path = locator.get_path(vid, frame)
    # path = <result_root>/P01_01_0003/frame_%10d.jpg
    """
    def __init__(self,
                 result_root,
                 cache_path,
                 verbose=False):
        self.result_root = Path(result_root)
        self.cache_path = to_absolute_path(cache_path)
        self._load_index()
        self.verbose = verbose

    def _load_index(self):
        # cache_path = osp.join('.cache', 'pair_index.pkl')
        if not osp.exists(self.cache_path):
            os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
            print("First time run, generating index...")
            _all_full_frames, _all_folders = self._build_index(
                self.result_root)
            with open(self.cache_path, 'wb') as fp:
                pickle.dump((_all_full_frames, _all_folders), fp)
            print("Index saved to", self.cache_path)

        with open(self.cache_path, 'rb') as fp:
            self._all_full_frames, self._all_folders = pickle.load(fp)

    def _build_index(self, result_root):

        def generate_pair_infos(root):
            pair_dir = os.listdir(root)
            dir_infos = []
            for d in tqdm.tqdm(pair_dir):
                l = sorted(os.listdir(osp.join(root, d)))
                mn, mx = l[0], l[-1]
                mn = int(re.search('\d{10}', mn)[0])
                mx = int(re.search('\d{10}', mx)[0])
                dir_infos.append( (d, mn, mx) )
            def func(l):
                x, _, _ = l
                a, b, c = x.split('_')
                a = a[1:]
                a = int(a)
                b = int(b)
                c = int(c)
                return a*1e7 + b*1e4 + c
            pair_infos = sorted(dir_infos, key=func)
            return pair_infos

        pair_infos = generate_pair_infos(result_root)  # pair_infos[i] = ['P01_01_0003', '123', '345']

        _all_full_frames = []
        _all_folders = []
        for folder, st, ed in pair_infos:
            min_frame = int(st)
            index = self._hash(folder, min_frame)
            _all_full_frames.append(index)
            _all_folders.append(folder)

        _all_full_frames = np.int64(_all_full_frames)
        sort_idx = np.argsort(_all_full_frames)
        _all_full_frames = _all_full_frames[sort_idx]
        _all_folders = np.asarray(_all_folders)[sort_idx]
        return _all_full_frames, _all_folders

    @staticmethod
    def _hash(vid: str, frame: int):
        pid, sub = vid.split('_')[:2]
        pid = pid[1:]
        op1, op2, op3 = map(int, (pid, sub, frame))
        index = op1 * int(1e15) + op2 * int(1e12) + op3
        return index

    def locate(self, vid, frame) -> Union[str, None]:
        """
        Returns: a str in DAVIS folder format: {vid}_{%4d}
            e.g P11_16_0107
        """
        query = self._hash(vid, frame)
        loc = bisect.bisect_right(self._all_full_frames, query)
        if loc == 0:
            return None
        r = self._all_folders[loc-1]
        r_vid = '_'.join(r.split('_')[:2])
        if vid != r_vid:
            if self.verbose:
                print(f"folder for {vid} not found")
            return None
        frames = map(
            lambda x: int(re.search('[0-9]{10}', x).group(0)),
            os.listdir(self.result_root/r))
        if max(frames) < frame:
            if self.verbose:
                print(f"Not found in {r}")
            return None
        return r

    def get_path(self, vid, frame):
        folder = self.locate(vid, frame)
        if folder is None:
            return None
        fname = f"{vid}_frame_{frame:010d}.jpg"
        fname = self.result_root/folder/fname
        return fname


class ImageLocator(PairLocator):
    def __init__(self, 
                 result_root='/home/skynet/Zhifan/data/visor-dense/480p',
                 cache_path='.cache/image_pair_index.pkl'):
        super().__init__(
            result_root=result_root,
            cache_path=cache_path)

    def get_path(self, vid, frame):
        folder = self.locate(vid, frame)
        if folder is None:
            return None
        fname = f"{vid}_frame_{frame:010d}.jpg"
        fname = self.result_root/folder/fname
        return fname


class UnfilteredMaskLocator(PairLocator):
    def __init__(self, 
                 result_root='/home/skynet/Zhifan/data/visor-dense/unfiltered_interpolations',
                 cache_path='.cache/mask_pair_index.pkl'):
        super().__init__(
            result_root=result_root,
            cache_path=cache_path)

    def get_path(self, vid, frame):
        folder = self.locate(vid, frame)
        if folder is None:
            return None
        fname = f"{vid}_frame_{frame:010d}.png"
        fname = self.result_root/folder/fname
        return fname


if __name__ == '__main__':
    """
    class LocatorExt(PairLocator):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            pair_info = kwargs.pop('pair_info', None)
            pair_info = io.read_txt('/media/skynet/DATA/Datasets/visor-dense/interpolations_pair_infos.txt')
            self.pair_info = {v[0]: (int(v[1]), int(v[2])) for v in pair_info}

        def folder_min_max(self, vid, frame) -> tuple:
            folder = self.locate(vid, frame)
            return self.pair_info[folder]
    """
    image_locator = ImageLocator()
    print( image_locator.get_path('P01_01', 28801) )

    mask_locator = UnfilteredMaskLocator()
    print( mask_locator.get_path('P01_01', 28801) )
