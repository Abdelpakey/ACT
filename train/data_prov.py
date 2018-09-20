import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import json
import pickle

# from src.visualization import show_frame_once

# extension to get the regions of both current and future(lookahead) frames
class ILSVRCDataset(object):
    def __init__(self, data_path, data_home):
        with open(data_path, 'r') as f:
          self.dataset = json.load(f)
        self.img_dir = data_home
        self.n_seq = len(self.dataset)
        self.seq_names = []
        self.gt = []
        self.ranges = []
        for i in range(self.n_seq):
            self.seq_names.append(self.dataset[i]['seq_name'])
        for i in range(self.n_seq):
            self.gt.append(np.array(self.dataset[i]['gt']))
        for i in range(self.n_seq):
            self.ranges.append((self.dataset[i]['start_frame'], self.dataset[i]['end_frame']))

        self.lookahead = 20

    def __iter__(self):
        return self

    def __next__(self):
        frame_list = []
        ground_list = []

        seq_id = np.random.randint(self.n_seq)
        start_frame = self.ranges[seq_id][0]
        end_frame = self.ranges[seq_id][1]
        start_idx = np.random.randint(max(1, end_frame-start_frame-self.lookahead*2))
        end_idx = min((start_idx + self.lookahead + np.random.randint(self.lookahead)), end_frame - start_frame + 1)
        # print start_frame, end_frame, start_idx, end_idx
        for idx in range(start_idx, end_idx):
            try:
                bbox = np.copy(self.gt[seq_id][idx])
            except:
                a = 1
            img_path = self.img_dir + ('/%s/%06d.JPEG'%(self.seq_names[seq_id], idx + start_frame - 1))
            frame_list.append(img_path)
            ground_list.append(bbox)

            # show_frame_once(np.array(Image.open(img_path).convert('RGB')), bbox, 2)
        return frame_list, ground_list, end_idx - start_idx


    next = __next__


