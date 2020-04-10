#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)
# Mostly based on the code written by Tinghui Zhou:
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class KtiitRawS(Dataset):
    def __init__(self, cfg):
        super(KtiitRawS, self).__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))

        test_scene_file = os.path.join(dir_path, 'splits/test_scene_eigen.txt')
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]

        self.dataset_dir = cfg.dataset_dir
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width
        self.img_ext = cfg.img_ext
        self.mode = cfg.mode
        self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29',
                          '2011_09_30', '2011_10_03']
        self.use_gt = False

        self.collect_train_frames()

    def collect_train_frames(self):
        all_frames = []
        all_depths = []
        for date in self.date_list:
            drive_set = os.listdir(os.path.join(self.dataset_dir, 'rawdata/', date))
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, 'rawdata', date, dr)
                if os.path.isdir(drive_dir):
                    if dr[:-5] in self.test_scenes:
                        continue
                    for cam in self.cam_ids:
                        gt_dir = os.path.join(self.dataset_dir, self.mode, dr,
                                              'proj_depth/groundtruth/image_{}'.format(cam))
                        if os.path.exists(gt_dir):
                            frames = sorted(os.listdir(gt_dir))
                            for frame in frames:
                                all_depths.append(dr + ' ' + cam + ' ' + frame)
                                all_frames.append(dr + ' ' + cam + ' ' + frame)

        self.train_frames = all_frames
        self.train_depths = all_depths

    def load_single_example(self, frames, tgt_idx):
        tgt_drive, tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')
        tgt_date = tgt_drive[:10]
        raw_image = self.load_image_raw(tgt_date, tgt_drive, tgt_cid, tgt_frame_id)
        zoom_y = self.img_height / raw_image.size[0]
        zoom_x = self.img_width / raw_image.size[1]
        intrinsics = self.load_intrinsics_raw(tgt_date, tgt_cid)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        cur_img = raw_image.resize((self.img_height, self.img_width), Image.NEAREST)

        raw_dmap = self.load_gt_raw(tgt_drive, tgt_frame_id)
        cur_dmap = raw_dmap.resize((self.img_height, self.img_width), Image.NEAREST)

        example = {}
        example['image'] = cur_img
        example['dmap'] = cur_dmap
        example['intrinsics'] = intrinsics
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/'
        example['file_name'] = tgt_frame_id

        return example

    def load_gt_raw(self, tgt_drive, tgt_frame_id):
        dmap_path = '{}/{}/{}/proj_depth/groundtruth/image_02/{}'. \
            format(self.dataset_dir, self.mode, tgt_drive, tgt_frame_id)
        dmap_raw = Image.open(dmap_path).convert('RGB')

        return dmap_raw

    def load_image_raw(self, date, drive, cam_id, frame):
        # data: 2011_09_26
        # drive: 2011_09_26_XXX
        # cam_id: 02/03
        # frame: 00000005
        im_path = os.path.join(self.dataset_dir, 'rawdata', date, drive, 'image_{}/data/'.format(cam_id), frame)
        im = Image.open(im_path).convert('RGB')
        return im

    def load_intrinsics_raw(self, date, cid):
        calib_file = os.path.join(self.dataset_dir, 'rawdata', date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        intrinsics = P_rect[:3, :3]
        return intrinsics

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
            return data

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out


    def __len__(self):
        return len(self.train_frames)

    def __getitem__(self, item):
        single_example = self.load_single_example(self.train_frames, item)

        return single_example


if __name__ == "__main__":
    from mmcv import ConfigDict
    cfg = dict(
        data=dict(
            dataset_dir='/data4/kitti_raw/',
            split='eigen',
            img_height=128,
            img_width=416,
            img_ext='.png',
            mode='train'
        )
    )
    cfg = ConfigDict(cfg)

    kitti_raw = KtiitRawS(cfg.data)
    sample = kitti_raw[0]

    print(kitti_raw.__len__())


    print("*")



