import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import utils.model_utils as mutils

"""Class based model"""
# Build the neural network, expand on top of nn.Module
class QuizDNN_Net(nn.Module):
    def __init__(self, drop_val=0):
        super(QuizDNN_Net, self).__init__()

        self.conv_blk1_l1 = mutils.Conv2d_BasicBlock(inC=3, outC=8, ksize=(3,3), padding=1, drop_val=drop_val) 	#Output: 8X32X32, Jin=1, GRF: 3X3
        self.conv_blk1_l2 = mutils.Conv2d_BasicBlock(inC=11, outC=11, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 11X32X32, Jin=1, GRF: 5X5
        self.maxpool1 = nn.MaxPool2d(2, 2)																		#Output: 11X16X16, Jin=1, Jout=2, GRF: 6X6

        self.conv_blk2_l1 = mutils.Conv2d_BasicBlock(inC=22, outC=22, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 22X16X16, Jin=2, GRF: 10X10
        self.conv_blk2_l2 = mutils.Conv2d_BasicBlock(inC=44, outC=44, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 44X16X16, Jin=2, GRF: 14X14
        self.conv_blk2_l3 = mutils.Conv2d_BasicBlock(inC=88, outC=88, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 88X16X16, Jin=2, GRF: 18X18
        self.maxpool2 = nn.MaxPool2d(2, 2)																		#Output: 88X8X8, Jin=2, Jout=4, GRF: 20X20

        self.conv_blk3_l1 = mutils.Conv2d_BasicBlock(inC=154, outC=154, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 154X4X4, Jin=4, GRF: 28X28
        self.conv_blk3_l2 = mutils.Conv2d_BasicBlock(inC=308, outC=308, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 308X4X4, Jin=4, GRF: 34X34
        self.conv_blk3_l3 = mutils.Conv2d_BasicBlock(inC=616, outC=616, ksize=(3,3), padding=1, drop_val=drop_val)#Output: 616X4X4, Jin=4, GRF: 42X42
        
        # Output Block
        self.output_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),                                                            #Output: 616X1X1, Jin=8, GRF: 54X54
            nn.Conv2d(in_channels=616, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),  #Output: 10X1X1, combining to 10 channels as we need 10 classes for predictions
            # no ReLU for last covv layer
            # No Batch Normalization
            # No Dropout
        ) # output_size = 1

    def forward(self, x1):
	
		# block-1
        x2 = self.conv_blk1_l1(x1) 

        x1_x2 = torch.cat([x1, x2], dim=1) 
        x3 = self.conv_blk1_l2(x1_x2)

        x1_x2_x3 = torch.cat([x1_x2, x3], dim=1)
        x4 = self.maxpool1(x1_x2_x3) 
        
		# block-2
        x5 = self.conv_blk2_l1(x4) 

        x4_x5 = torch.cat([x4, x5], dim=1) 
        x6 = self.conv_blk2_l2(x4_x5) 

        x4_x5_x6 = torch.cat([x4_x5, x6], dim=1)
        x7 = self.conv_blk2_l3(x4_x5_x6) 

        x5_x6 = torch.cat([x5, x6], dim=1)
        x5_x6_x7 = torch.cat([x5_x6, x7], dim=1)
        x8 = self.maxpool2(x5_x6_x7) 
        
		# block-3
        x9 = self.conv_blk3_l1(x8) 
        x8_x9 = torch.cat([x8, x8], dim=1)
        x10 = self.conv_blk3_l2(x8_x9) 

        x8_x9_x10 = torch.cat([x8_x9, x10], dim=1)
        x11 = self.conv_blk3_l3(x8_x9_x10)

        # output 
        x = self.output_block(x11) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)
		
