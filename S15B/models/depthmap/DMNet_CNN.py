import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import utils.cnn_utils as cutils
from utils.TimeMemoryProfile import MemoryUsage, TimeProfile
from torch.utils.tensorboard import SummaryWriter

#Dependecies: user need to load the necessary extension
'''
if pkgutil.find_loader("tensorboard") is None:
  !pip install tensorboard
  
#Load the TensorBoard extension
%load_ext tensorboard
'''

"""Class based model"""
# Build the neural network, expand on top of nn.Module
class CNNDepthNet(nn.Module):
    def __init__(self, drop_val=0):
        super(CNNDepthNet, self).__init__()

        self.conv_blk = nn.Sequential(
            # input layer
            cutils.Conv2d_BasicBlock(inC=6, outC=16, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=32, outC=48, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=48, outC=64, ksize=(3,3), padding=1, drop_val=drop_val)
        )

        self.combine = nn.Sequential(           
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), bias=False)
        )

        self.conv_blk_mask = nn.Sequential(
            cutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val)
        )

        self.conv_blk_depth = nn.Sequential(
            cutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val),
            cutils.Conv2d_BasicBlock(inC=64, outC=128, ksize=(3,3), padding=1, drop_val=drop_val)
        )

        # Last layer for mask prediction
        self.conv_final_mask = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), bias=False)
        )

        # Last layer for depth prediction
        self.conv_final_depth = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), bias=False)
        )

    def forward(self, x):
       # Here input is of 6 channels, mask and depth channels: 3+3=6

        # common conv block
        x = self.conv_blk(x)
        x = self.combine(x)

        # dedicated conv blk for mask prediction
        m1 = self.conv_blk_mask(x) # convolution block-4

        # last conv layer for mask prediction - ground truth mask is of 1 channel
        m2 = self.conv_final_mask(m1) 

        # dedicated conv blk for depth prediction
        d1 = self.conv_blk_depth(x) # convolution block-4

        # last conv layer for depth prediction - ground truth depth is of 1 channel
        d2 = self.conv_final_depth(d1) 

        return m2, d2
