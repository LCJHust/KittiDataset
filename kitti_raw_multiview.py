#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_hust@126.com)

# Mostly based on the code written by Tinghui Zhou:
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py
import numpy as np
from glob import glob
import os
import scipy.misc
from PIL import Image
import torch
from torch.utils.data import Dataset


class KittiRawBase(Dataset):
    def __init__(self, cfg):
        super(KittiRawBase, self).__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))

        ## Remove the frames in eigen test
        test_scene_file = os.path.join(dir_path, 'splits/test_scene_eigen.txt')
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = cfg.dataset_dir
        self.img_height = cfg.img_height
        self.img_width = cfg.img_width
        self.seq_length = cfg.seq_length
        self.half_offset = int((self.seq_length - 1) / 2)
        self.cam_ids = ['02']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29',
                          '2011_09_30', '2011_10_03']
        self.img_ext = cfg.img_ext

        static_frames_file = dir_path + '/splits/static_frames.txt'
        self.collect_static_frames(static_frames_file)
        self.collect_train_frames()

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = []
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            for cid in self.cam_ids:
                self.static_frames.append(drive + ' ' + cid + ' ' + curr_fid)

    def collect_train_frames(self):
        all_frames = []
        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir + date + '/')
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)
                if os.path.isdir(drive_dir):
                    if dr[:-5] in self.test_scenes:
                        continue
                    for cam in self.cam_ids:
                        img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
                        frames = sorted(os.listdir(img_dir))
                        for frame in frames[self.half_offset: -self.half_offset]:
                            all_frames.append(dr + ' ' + cam + ' ' + frame)

            for s in self.static_frames:
                try:
                    all_frames.remove(s)
                except:
                    pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        # return img_list = [-2, -1, 0, 1, 2]
        half_offset = int((seq_length - 1) / 2)
        image_seq = []
        file_seq = []
        for o in range( 2 * half_offset + 1):
            curr_idx = tgt_idx + o
            curr_drive, curr_cid, curr_frame_id = frames[curr_idx].split(' ')
            curr_date = curr_drive[:10]
            curr_img = self.load_image_raw(curr_date, curr_drive, curr_cid, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height / curr_img.size[0]
                zoom_x = self.img_width / curr_img.size[1]
            curr_img = curr_img.resize((self.img_height, self.img_width), Image.NEAREST)
            image_seq.append(curr_img)
            file_seq.append(os.path.join(curr_drive, curr_cid, curr_frame_id))

        return file_seq, zoom_x, zoom_y

    def load_sequence_example(self, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(self.train_frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_cid, tgt_frame_id = self.train_frames[tgt_idx].split(' ')
        tgt_date = tgt_drive[:10]
        intrinsics = self.load_intrinsics_raw(tgt_date, tgt_cid)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/'
        example['file_name'] = tgt_frame_id
        return example

    def load_single_example(self, frames, tgt_idx):
        tgt_drive, tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')
        tgt_date = tgt_drive[:10]
        intrinsics = self.load_intrinsics_raw(tgt_date, tgt_cid)
        example = {}
        example['intrinsics'] = intrinsics
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/'
        example['file_name'] = tgt_frame_id
        return example

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

    def load_image_raw(self, date, drive, cam_id, frame):
        # data: 2011_09_26
        # drive: 2011_09_26_XXX
        # cam_id: 02/03
        # frame: 00000005
        im_path = os.path.join(self.dataset_dir, date, drive, 'image_{}/data/'.format(cam_id), frame)
        im = Image.open(im_path).convert('RGB')
        return im

    def load_intrinsics_raw(self, date, cid):
        calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        intrinsics = P_rect[:3, :3]
        return intrinsics

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

        return self.load_sequence_example(item)



if __name__ == "__main__":
    from mmcv import ConfigDict
    cfg = dict(
        dataset_dir='/data4/kitti_raw/rawdata/',
        split='eigen',
        img_height=128,
        img_width=416,
        seq_length=3,
        img_ext='.png',
        remove_static=True

    )
    cfg = ConfigDict(cfg)

    kitti_raw = KittiRawBase(cfg)

    seq = kitti_raw[0]

    print("*")
