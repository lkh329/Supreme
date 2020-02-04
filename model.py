import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
class STB_Block(nn.Module):
    def __init__(self, in_features):
        super(STB_Block, self).__init__()
        stb_block =[nn.PReLU(),
                   #nn.ReLU(),
                   #Spatial-temporal Conv
                   nn.Conv3d(in_features, in_features, (1, 3, 3), 1, (0, 1, 1)),
                   nn.Conv3d(in_features, in_features, (3, 1, 1), 1, (1, 0, 0))]
        self.stb_block = nn.Sequential(*stb_block)
        
    def forward(self, x):
        return x+self.stb_block(x)
    
class Supreme(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16,
                 base_channels=64, img_width=32, img_height=32, zoom=2,depth=5,ext_flag=True):
        super(Supreme, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.zoom = zoom
        self.depth=depth
        self.ext_flag = ext_flag
        
        if ext_flag:
            self.embed_day = nn.Embedding(8, 2) # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3) # hour range [0, 23]
            self.embed_weather = nn.Embedding(26, 3) # determined by the type of weathers
           
            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, img_width * img_height),
                nn.ReLU(inplace=True)
            )
            #number of pixelshuffle = zoom/2
            if self.zoom==2:
                self.ext2hr = nn.Sequential(
                    nn.Conv2d(1, 4, 3, 1, 1),
                    nn.BatchNorm2d(4),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.ReLU(inplace=True),
                )
            elif self.zoom == 4:
                self.ext2hr = nn.Sequential(
                    nn.Conv2d(1, 4, 3, 1, 1),
                    nn.BatchNorm2d(4),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1, 4, 3, 1, 1),
                    nn.BatchNorm2d(4),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.ReLU(inplace=True),
                )
        if ext_flag:
            conv1_in = in_channels
            conv3_in = base_channels + 1
        else:
            conv1_in = in_channels
            conv3_in = base_channels
            
        # input conv 
        self.conv1 = nn.Sequential(
            nn.Conv3d( conv1_in, base_channels, (3,9,9), stride = 1, padding=(1,4,4)),
            nn.PReLU()
            #nn.ReLU(inplace=True)
        )
        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, out_channels, 9, 1, 4),
            nn.PReLU()
            #nn.ReLU(inplace=True)
        )
        

        # Spatial-temporal Residual blocks
        stb_blocks = []
        for _ in range(n_residual_blocks):
            stb_blocks.append(STB_Block(base_channels))
        self.stb_blocks = nn.Sequential(*stb_blocks)
        
        # Second conv layer post ST residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels,(3,5,5), stride = 1, padding=(1,2,2)))
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels,(self.depth,5,5), stride = 1, padding=(0,2,2)))

        # Reconstruction & upsampling
        upsampling = []
        for out_features in range(int(zoom/2)):
            upsampling += [nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)
        self.adpooling = nn.AdaptiveAvgPool3d((1,None,None))
        self.oriupsample = nn.Upsample(
            scale_factor=zoom, mode='nearest')
        
    def forward(self, x, ext):
        inp = x
        oriup = self.oriupsample(x[:,:,self.depth-1,:,:])
        if self.ext_flag:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width, self.img_height)
            lr_ext_out = torch.unsqueeze(ext_out.repeat(1,self.depth,1,1),1)

            
 
        out1 = self.conv1(inp)
        out2 = self.stb_blocks(out1)

        out = torch.add(out1, out2)
        out = torch.squeeze(self.adpooling(out),2)
        out = self.upsampling(out)
        
        # concatenation of external factor
        if self.ext_flag:
            ext_out = self.ext2hr(ext_out)
            out = self.conv3(torch.cat([out, ext_out], dim=1))
        else:
            out = self.conv3(out)
        #skip connection
        out+=oriup
    
        return out
