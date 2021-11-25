#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import json
import cv2
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data

def get_annotation_list(dataset_dir):
    image_dir  = os.path.join(dataset_dir, 'train_images')
    # read csv
    df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    anno_list = []
    for i, row in tqdm(df.iterrows()):
        video_path = "video_{}".format(row['video_id']) + "/{}.jpg".format(row['video_frame'])
        loc = ast.literal_eval(row['annotations'])
        anno_list.append({'img_path':os.path.join(image_dir, video_path), 'loc':loc})
    return anno_list

class TF_GBR_Dataset(data.Dataset):
    def __init__(self, dataset_dir, resize_sz = (224, 224), transform=False):
        super(TF_GBR_Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.data = get_annotation_list(self.dataset_dir)
        self.resize_sz = resize_sz
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ndx):
        anno_data = self.data[ndx]
        # read image
        img = cv2.imread(anno_data['img_path'], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,self.resize_sz)
        img = np.array(img, dtype=np.float32)
        # normalize
        img = img / 255
        img = torch.from_numpy(img).permute(2, 0, 1)    #(H, W, C) -> (C, H, W)

        if self.transform:
            pass

        # get location data
        loc_list = []
        for d in anno_data['loc']:
            loc_list.append(torch.tensor([d['x'], d['y'], d['width'], d['height']], dtype=torch.float32))

        if len(loc_list) > 0:
            loc_list = torch.stack(loc_list)
        else:
            loc_list = torch.tensor([], dtype=torch.float32)

        return img, loc_list

if __name__ == "__main__":
    dataset_dir = "/home/take/fun/dataset/kaggle/tensorflow-great-barrier-reef"
    dataset = TF_GBR_Dataset(dataset_dir)
    img, loc_list = dataset.__getitem__(100)
    print(type(img))
    print(type(loc_list))
