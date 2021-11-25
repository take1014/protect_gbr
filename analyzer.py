#!usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import cv2
import ast
import pandas as pd
from tqdm import tqdm
from dataset import get_annotation_list

def draw_bounding_box(img, loc_list):
    for loc in loc_list:
        img = cv2.line(img, (loc['x'], loc['y']), (loc['x']+loc['width'], loc['y']),                             (0, 0, 255), 2)
        img = cv2.line(img, (loc['x'], loc['y']), (loc['x'], loc['y']+loc['height']),                            (0, 0, 255), 2)
        img = cv2.line(img, (loc['x']+loc['width'], loc['y']), (loc['x']+loc['width'], loc['y']+loc['height']),  (0, 0, 255), 2)
        img = cv2.line(img, (loc['x'], loc['y']+loc['height']), (loc['x']+loc['width'], loc['y']+loc['height']), (0, 0, 255), 2)
    return img

def calc_bbox_rate(df_anno):
    num_bbox = df_anno.apply(lambda x: len(ast.literal_eval(x)))
    return (num_bbox>0).value_counts()/len(df)*100

if __name__ == '__main__':
    dataset_dir = "/home/take/fun/dataset/kaggle/tensorflow-great-barrier-reef"

    # read csv
    df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))

    # calc bounding box rate
    data = calc_bbox_rate(df['annotations'])
    print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

    # get annotation's list
    anno_list = get_annotation_list(dataset_dir)

    if not os.path.exists('./bd_img'):
        os.mkdir('./bd_img')

    for i, anno in tqdm(enumerate(anno_list)):
        image_path = anno['img_path']
        loc_list   = anno['loc']
        print(image_path)
        image_file_name = os.path.basename(image_path)
        save_folder_path     = os.path.join('./bd_img', image_path.split('/')[-2])
        if not os.path.exists(save_folder_path):
            os.mkdir(os.path.join(save_folder_path))

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # draw bounding box on image
        draw_bounding_box(img, loc_list)

        cv2.imwrite(os.path.join(save_folder_path, '{}.jpg'.format(str(i).zfill(5))), img)
