#!/usr/bin/env python
# -*- coding: utf-8 -*-
from homan.datasets.core50 import Core50
from homan.datasets.epic import Epic
from homan.datasets.ho3d import HO3D


def get_dataset(
    dataset,
    split="train",
    frame_nb=10,
    box_mode="track",
    load_img=True,
    chunk_step=4,
    use_cache=True,
    dataset_mode="chunk",
    epic_hdf5_root=None,
    epic_root="",
    epic_valid_path="",
):
    if dataset == "ho3d":
        image_size = 640
        dataset = HO3D(
            split=split,
            frame_nb=frame_nb,
            box_mode=box_mode,
            load_img=load_img,
            # mode="vid",
            mode=dataset_mode,
            chunk_step=chunk_step,
            use_cache=use_cache)
    elif dataset == "epic":
        image_size = 640
        dataset = Epic(mode=dataset_mode,
                       frame_nb=frame_nb,
                       frame_step=1, # TODO(zhifan, Mar 06): change this?
                       use_cache=use_cache,
                       hdf5_root=epic_hdf5_root,
                       epic_root=epic_root,
                       valid_vids_path=epic_valid_path)
                       
    elif dataset == "core50":
        image_size = 350
        dataset = Core50(frame_nb=frame_nb,
                         track=False,
                         load_img=load_img,
                         chunk_step=chunk_step,
                         use_cache=use_cache,
                         mode=dataset_mode)
    else:
        raise ValueError(
            f"{dataset} not in [ho3d|contactpose|core50]")
    return dataset, image_size
