import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import utils.model_utils as mutils

"""Class based model"""
# Build the neural network, expand on top of nn.Module
class Cifar10Net(nn.Module):
    def __init__(self, drop_val=0):
        super(Cifar10Net, self).__init__()

        # Convolution block-1
        self.conv_blk1 = nn.Sequential(
            # input layer
            mutils.Conv2d_BasicBlock(inC=3, outC=16, ksize=(3,3), padding=1, drop_val=drop_val),             #Output: 16X32X32, Jin=1, GRF: 3X3
            mutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),            #Output: 32X32X32, Jin=1, GRF: 5X5
            mutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val)             #Output: 64X32X32, Jin=1, GRF: 7X7
        )

        # Transition Layer for Convolution block-1
        # Maxpooling followed by 1X1 conv
        self.conv_blk1_transition = mutils.Conv2d_TransistionBlock(64,16)                                    #Output: 16X16X16, Jin=1, Jout=2, GRF: 8X8

        # Convolution block-2
        self.conv_blk2 = nn.Sequential(
            mutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val),#Output: 32X14X14, Jin=2, GRF: 16X16
            mutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val) #Output: 64X12X12, Jin=2, GRF: 22X22
        )

        # Transition Layer for Convolution block-2
        self.conv_blk2_transition = mutils.Conv2d_TransistionBlock(64,16)                                    #Output: 16X6X6, Jin=2, Jout=4, GRF: 24X24

        # Convolution block-3
        self.conv_blk3 = nn.Sequential(
            mutils.Conv2d_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),			 #Output: 32X6X6, Jin=4, GRF: 32X32
            mutils.Conv2d_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val) 			 #Output: 64X6X6, Jin=4, GRF: 40X40
        )

        # Transition Layer for Convolution block-2
        self.conv_blk3_transition = mutils.Conv2d_TransistionBlock(64,16)                                    #Output: 16X3X3, Jin=4, Jout=8, GRF: 44X44

        # Convolution block-4
        self.conv_blk4 = nn.Sequential(
            mutils.Conv2d_Seperable_BasicBlock(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),  #Output: 32X3X3, Jin=8, GRF: 60X60
            mutils.Conv2d_Seperable_BasicBlock(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val),  #Output: 64X3X3, Jin=8, GRF: 76X76
        )

        # Output Block
        self.output_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),                                                            #Output: 64X1X1, Jin=8, GRF: 92X92
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),  #Output: 10X1X1, combining to 10 channels as we need 10 classes for predictions
            # no ReLU for last covv layer
            # No Batch Normalization
            # No Dropout
        ) # output_size = 1

    def forward(self, x):

        x = self.conv_blk1(x) # convolution block-1
        x = self.conv_blk1_transition(x)

        x = self.conv_blk2(x) # convolution block-2
        x = self.conv_blk2_transition(x)

        x = self.conv_blk3(x) # convolution block-3
        x = self.conv_blk3_transition(x)

        x = self.conv_blk4(x) # convolution block-4

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)

"""Function Based-Model"""

# Build the neural network, expand on top of nn.Module
# BasicBlock are implemented as functions within class
class Cifar10NetFxnBased(nn.Module):
    def __init__(self, drop_val=0):
        super(Cifar10NetFxnBased, self).__init__()

        # Convolution block-1
        self.conv_blk1 = nn.Sequential(
            # input layer
            self.conv2d_basicBlock_fxn(inC=3, outC=16, ksize=(3,3), padding=1, drop_val=drop_val),             #Output: 16X32X32, Jin=1, GRF: 3X3
            self.conv2d_basicBlock_fxn(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),            #Output: 32X32X32, Jin=1, GRF: 5X5
            self.conv2d_basicBlock_fxn(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val)             #Output: 64X32X32, Jin=1, GRF: 7X7
        )

        # Transition Layer for Convolution block-1
        # Maxpooling followed by 1X1 conv
        self.conv_blk1_transition = self.conv2d_transistion_blk_fxn(64,16)                                     #Output: 16X16X16, Jin=1, Jout=2, GRF: 8X8

        # Convolution block-2
        self.conv_blk2 = nn.Sequential(
            self.conv2d_basicBlock_fxn(inC=16, outC=32, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val), #Output: 32X14X14, Jin=2, GRF: 16X16
            self.conv2d_basicBlock_fxn(inC=32, outC=64, ksize=(3,3), padding=1, dilation=2, drop_val=drop_val)  #Output: 64X12X12, Jin=2, GRF: 22X22
        )

        # Transition Layer for Convolution block-2
        self.conv_blk2_transition = self.conv2d_transistion_blk_fxn(64,16)                                      #Output: 16X6X6, Jin=2, Jout=4, GRF: 24X24

        # Convolution block-3
        self.conv_blk3 = nn.Sequential(
            self.conv2d_basicBlock_fxn(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),				#Output: 32X6X6, Jin=4, GRF: 32X32
            self.conv2d_basicBlock_fxn(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val) 				#Output: 64X6X6, Jin=4, GRF: 40X40
        )

        # Transition Layer for Convolution block-3
        self.conv_blk3_transition = self.conv2d_transistion_blk_fxn(64,16)                                      #Output: 16X3X3, Jin=4, Jout=8, GRF: 44X44

        # Convolution block-4
        self.conv_blk4 = nn.Sequential(
            self.conv2d_seperable_basicBlock_fxn(inC=16, outC=32, ksize=(3,3), padding=1, drop_val=drop_val),   #Output: 32X3X3, Jin=8, GRF: 60X60
            self.conv2d_seperable_basicBlock_fxn(inC=32, outC=64, ksize=(3,3), padding=1, drop_val=drop_val),   #Output: 64X3X3, Jin=8, GRF: 76X76
        )

        # Output Block
        self.output_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=3),                                                            #Output: 64X1X1, Jin=8, GRF: 92X92
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),  #Output: 10X1X1, combining to 10 channels as we need 10 classes for predictions
            # no ReLU for last covv layer
            # No Batch Normalization
            # No Dropout
        ) # output_size = 1
    
    def conv2d_Seperable_fxn(self, inC, outC, ksize, padding=0):

        return nn.Sequential(
            nn.Conv2d(in_channels=inC, out_channels=inC, groups=inC, kernel_size=ksize, padding=padding,  bias=False), # depth convolution
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1,1), bias=False) # Pointwise convolution
        )

    def conv2d_seperable_basicBlock_fxn(self, inC, outC, ksize, padding=0, drop_val=0):

        basic = nn.Sequential(
            self.conv2d_Seperable_fxn(inC, outC, ksize, padding), # depth convolution + Pointwise convolution
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        if(drop_val!=0):
          basic = nn.Sequential(
              basic,
              nn.Dropout(drop_val)

          )
        
        return basic

    def conv2d_basicBlock_fxn(self, inC, outC, ksize, padding=0, dilation=1, drop_val=0):

        basic = nn.Sequential(
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=ksize, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        if(drop_val!=0):
          basic = nn.Sequential(
              basic,
              nn.Dropout(drop_val)

          )
        
        return basic

    def conv2d_transistion_blk_fxn(self, inC, outC):

        return nn.Sequential(
            nn.MaxPool2d(2, 2),                                                           #Output: 16X16X16, Jin=1, GRF: 8X8
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1, 1), bias=False),  #Output: 8X13X13 , Jin=2, GRF: 8X8 (combining channels)
        )

    def forward(self, x):

        x = self.conv_blk1(x) # convolution block-1
        x = self.conv_blk1_transition(x)

        x = self.conv_blk2(x) # convolution block-2
        x = self.conv_blk2_transition(x)

        x = self.conv_blk3(x) # convolution block-3
        x = self.conv_blk3_transition(x)

        x = self.conv_blk4(x) # convolution block-4

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)
