import os
import io
import numpy as np
from PIL import Image

import h5py

import tqdm
import pickle
import pandas as pd
from pathlib import Path


class EpicHdf5Reader(object):
    
    def __init__(self, hdf5_root):
        """

        Args:
            hdf5_root (str): path to 'epic_hdf5_rgb_frames'
        """
        hds = os.listdir(hdf5_root)
        self.video_id_map = {}
        for hd in hds:
            video_id = hd.replace('.hdf5', '')
            hd_file = os.path.join(hdf5_root, hd)
            self.video_id_map[video_id] = h5py.File(hd_file, 'r')

    def read_frame_pil(self, video_id, frame_idx):
        """

        Args:
            video_id (str)
            frame_idx (str): frame_{:010d}
        
        Returns:
            PIL.Image
        """
        d =  self.video_id_map[video_id][frame_idx]
        return Image.open(io.BytesIO(np.array(d)))

    def read_frame_np(self, video_id, frame_idx):
        """

        Args:
            video_id (str)
            frame_idx (str): "frame_{:010d}"
        
        Returns:
            np.ndarray
        """
        d =  self.video_id_map[video_id][frame_idx]
        return np.asarray(Image.open(io.BytesIO(np.array(d))))


def make_hdf5_dataset(
    valid_vids_path='/home/deepthought/Zhifan/allVideos.xlsx',
    epic_rgb_root=Path('/home/skynet/Zhifan/data/epic/rgb_root/'),
    all_annotations='/home/skynet/Zhifan/data/epic/EPIC_100_train.pkl',
    save_dir='epic_hdf5_root'
):

    with open(all_annotations, 'rb') as fp:
        annot_df = pickle.load(fp)

    # annot_df = pd.read_csv(all_annotations)
    valid_vids = pd.read_excel(valid_vids_path)

    nouns=[
                "can",
                "cup",
                "plate",
                "bottle",
                # "pitcher",
                # "jug",
                # "phone",
            ]

    annot_df = annot_df[annot_df.video_id.isin(valid_vids['Unnamed: 0'])]
    annot_df = annot_df[annot_df.noun.isin(nouns)]

    track_padding = 51  # make sure to include more frames

    for _, annot in tqdm.tqdm(annot_df.iterrows(), total=len(annot_df)):
        video_id = annot.video_id
        pid = video_id.split('_')[0]
        vid_root = epic_rgb_root/pid/video_id
        start_frame, end_frame = annot['start_frame'], annot['stop_frame']
        with h5py.File(f'./{save_dir}/{video_id}.hdf5', 'a') as f:
            for frame_idx in tqdm.tqdm(
                range(start_frame-track_padding, end_frame+track_padding)):
                frame = f'frame_{frame_idx:010d}'
                if frame in f.keys():
                    continue
                imgpath = vid_root/f'{frame}.jpg'
                with open(imgpath, 'rb') as img_f:
                    img_bin = img_f.read()
                    f.create_dataset(frame, data=np.asarray(img_bin))


if __name__ == '__main__':
    make_hdf5_dataset()