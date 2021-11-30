#!/usr/bin/env python3
#-*- coding:utf-8 -*-

class Config():
    def __init__(self):
        self.work_dir     = "/home/take/fun/dataset/kaggle/tensorflow-great-barrier-reef"
        self.img_width    = 1280
        self.img_height   = 720
        self.img_channels = 3
        self.resize_sz   = (32*16, 32*9)        #(W, H)
        self.batch_sz     = 32

if __name__ == "__main__":
    config = Config()
    print(config.resize_sz)

