import os
import os.path as osp
import torch
import pandas as pd
import tqdm

import sys
sys.path.append('/home/skynet/Zhifan/repos/CPF')

from report.metrics import (
    to_scene, max_intersect_volume, get_meshes,compute_pen_depth,
    compute_hand_iou, compute_obj_iou
)


def main():
    root = '/home/skynet/Zhifan/homan-master/results/epic_visor2_nobeta/samples'
    videos = os.listdir(root)

    df = pd.read_csv('report/numbers/epic_visor2_no_beta.csv')
    visited = set(df['video'].tolist())
    need_videos = [v for v in videos if v not in visited]
    bad_videos = {
        'P06_108_29139_29395'
    }

    vid_keys = []
    hand_ious = []
    obj_ious = []
    pds = []
    ivs = []
    for video in tqdm.tqdm(need_videos):
        if video in visited:
            continue
        if video in bad_videos:
            continue
        fpath = osp.join(root, video, 'model.pth')
        if not os.path.exists(fpath):
            continue
        homan = torch.load(fpath)
        hiou = compute_hand_iou(homan)
        oiou = compute_obj_iou(homan)
        pend = compute_pen_depth(homan) # in mm
        iv = max_intersect_volume(homan, pitch=0.005, ret_all=False) # cm^3
        vid_keys.append(video)
        hand_ious.append(hiou)
        obj_ious.append(oiou)
        pds.append(pend)
        ivs.append(iv)
        visited.add(video)
        del homan
        torch.cuda.empty_cache()
    
    df = pd.DataFrame({
        'video': vid_keys,
        'hand_iou': hand_ious,
        'obj_iou': obj_ious,
        'pd': pds,
        'iv': ivs
    })
    df.to_csv('report/numbers/epic_visor2_no_beta.csv', index=False)


if __name__ == '__main__':
    main()