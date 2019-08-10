import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time

# we define Hour Glass network based on the paper
# Stacked Hourglass Networks for Human Pose Estimation
#       Alejandro Newell, Kaiyu Yang, and Jia Deng
# the code is adapted from
# https://github.com/umich-vl/pose-hg-train/blob/master/src/models/hg.lua


def conv3X3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)
# define the network
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, batchNorm_type=0, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # batchNorm_type 0 means batchnormalization
        #                1 means instance normalization
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, 1)
        self.conv2 = conv3X3(outplanes, outplanes, 1)
        if batchNorm_type == 0:
            self.bn1 = nn.BatchNorm2d(outplanes)
            self.bn2 = nn.BatchNorm2d(outplanes)
        else:
            self.bn1 = nn.InstanceNorm2d(outplanes)
            self.bn2 = nn.InstanceNorm2d(outplanes)
        
        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.inplanes != self.outplanes:
        		out += self.shortcuts(x)
        else:
        		out += x
        
        out = F.relu(out)
        return out

class HourglassBlock(nn.Module):
    '''
        define a basic block for hourglass neetwork
            ^-------------------------upper conv-------------------
            |                                                      |
            |                                                      V
        input------>downsample-->low1-->middle-->low2-->upsample-->+-->output
        NOTE about output:
            Since we need the lighting from the inner most layer, 
            let's also output the results from middel layer
    '''
    def __init__(self, inplane, mid_plane, middleNet, skipLayer=True):
        super(HourglassBlock, self).__init__()
        # upper branch
        self.skipLayer = True
        self.upper = BasicBlock(inplane, inplane, batchNorm_type=1)
        
        # lower branch
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.low1 = BasicBlock(inplane, mid_plane)
        self.middle = middleNet
        self.low2 = BasicBlock(mid_plane, inplane, batchNorm_type=1)

    def forward(self, x, light, count, skip_count):
        # we use count to indicate wich layer we are in
        # max_count indicates the from which layer, we would use skip connections
        out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        out_lower, out_middle = self.middle(out_lower, light, count+1, skip_count)
        out_lower = self.low2(out_lower)
        out_lower = self.upSample(out_lower)

        if count >= skip_count and self.skipLayer:
            # withSkip is true, then we use skip layer
            # easy for analysis
            out = out_lower + out_upper
        else:
            out = out_lower
            #out = out_upper
        return out, out_middle

class lightingNet(nn.Module):
    '''
        define lighting network
    '''
    def __init__(self, ncInput, ncOutput, ncMiddle):
        super(lightingNet, self).__init__()
        self.ncInput = ncInput
        self.ncOutput = ncOutput
        self.ncMiddle = ncMiddle

        # basic idea is to compute the average of the channel corresponding to lighting
        # using fully connected layers to get the lighting
        # then fully connected layers to get back to the output size

        self.predict_FC1 = nn.Conv2d(self.ncInput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.predict_relu1 = nn.PReLU()
        self.predict_FC2 = nn.Conv2d(self.ncMiddle, self.ncOutput, kernel_size=1, stride=1, bias=False)

        self.post_FC1 = nn.Conv2d(self.ncOutput,  self.ncMiddle, kernel_size=1, stride=1, bias=False)
        self.post_relu1 = nn.PReLU()
        self.post_FC2 = nn.Conv2d(self.ncMiddle, self.ncInput, kernel_size=1, stride=1, bias=False)
        self.post_relu2 = nn.ReLU()  # to be consistance with the original feature

    def forward(self, innerFeat, target_light, count, skip_count):
        x = innerFeat[:,0:self.ncInput,:,:] # lighting feature
        _, _, row, col = x.shape

        # predict lighting
        feat = x.mean(dim=(2,3), keepdim=True)
        light = self.predict_relu1(self.predict_FC1(feat))
        light = self.predict_FC2(light)

        # get back the feature space
        upFeat = self.post_relu1(self.post_FC1(target_light))
        upFeat = self.post_relu2(self.post_FC2(upFeat))
        upFeat = upFeat.repeat((1,1,row, col))
        innerFeat[:,0:self.ncInput,:,:] = upFeat
        return innerFeat, light


class HourglassNet(nn.Module):
    '''
    	basic idea: low layers are shared, upper layers are different	
    	            lighting should be estimated from the inner most layer
        NOTE: we split the bottle neck layer into albedo, normal and lighting
    '''
    def __init__(self, baseFilter = 16, gray=True):
        super(HourglassNet, self).__init__()

        self.ncLight = 27   # number of channels for input to lighting network
        self.baseFilter = baseFilter

        # number of channles for output of lighting network
        if gray:
            self.ncOutLight = 9  # gray: channel is 1
        else:
            self.ncOutLight = 27  # color: channel is 3

        self.ncPre = self.baseFilter  # number of channels for pre-convolution

        # number of channels 
        self.ncHG3 = self.baseFilter
        self.ncHG2 = 2*self.baseFilter
        self.ncHG1 = 4*self.baseFilter
        self.ncHG0 = 8*self.baseFilter + self.ncLight

        self.pre_conv = nn.Conv2d(1, self.ncPre, kernel_size=5, stride=1, padding=2)
        self.pre_bn = nn.BatchNorm2d(self.ncPre)

        self.light = lightingNet(self.ncLight, self.ncOutLight, 128)
        self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, self.light)
        self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.HG0)
        self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
        self.HG3 = HourglassBlock(self.ncPre, self.ncHG3, self.HG2)

        self.conv_1 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(self.ncPre) 
        self.conv_2 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.ncPre) 
        self.conv_3 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.ncPre)

        self.output = nn.Conv2d(self.ncPre, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, target_light, skip_count):
        feat = self.pre_conv(x)
        feat = F.relu(self.pre_bn(feat))
        # get the inner most features
        feat, out_light = self.HG3(feat, target_light, 0, skip_count)
        feat = F.relu(self.bn_1(self.conv_1(feat)))
        feat = F.relu(self.bn_2(self.conv_2(feat)))
        feat = F.relu(self.bn_3(self.conv_3(feat)))
        out_img = self.output(feat)
        out_img = torch.sigmoid(out_img)
        return out_img, out_light

if __name__ == '__main__':
    pass
