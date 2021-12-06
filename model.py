#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import torch
import torch.nn as nn

# TODO: Implement SSD model here

class PGBRModel(nn.Module):
    def __init__(self, in_channels=3):
        super(PGBRModel, self).__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    model = PGBRModel(in_channels=3)
    print(model)
