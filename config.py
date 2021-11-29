#!/usr/bin/env python3
#-*- coding:utf-8 -*-

class Config():
    def __init__(self):
        self.img_width    = 1280
        self.img_height   = 720
        self.img_channels = 3
        self.resize_CHW   = (3, 32*9, 32*16)

if __name__ == "__main__":
    config = Config()
    print(config.resize_CHW)
