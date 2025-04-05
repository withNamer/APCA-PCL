# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:21:36 2022

@author: loua2
"""

import functools

import torch
import torch.nn as nn
from utils.utils import img2seq, seq2img

class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.relu = nn.ReLU(inplace = True)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)  
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1) # 注意，conv不是一个1 * 1的东西，而是一个3 * 3的东西，所以很可能融合了一些不该融合的东西
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)
        

        return y_1
    
class conv_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)  
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 注意，conv不是一个1 * 1的东西，而是一个3 * 3的东西，所以很可能融合了一些不该融合的东西
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)
        

        return y_1
    
# From PyTorch internals
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_chans=4, embed_dim=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim * 2,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim * 2)
        else:
            self.norm = None
        self.conv = conv_(embed_dim * 2, embed_dim*2)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.conv(x)
        # x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

# class projectors(nn.Module):
#     def __init__(self, input_nc=4, ndf=8, norm_layer=nn.BatchNorm2d):
#         super(projectors, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv_1 = conv(input_nc, ndf)
#         self.conv_2 = conv(ndf, ndf*2)
#         self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
#     def forward(self, input):
#         x_0 = self.conv_1(input)
#         x_0 = self.pool(x_0)
#         x_out = self.conv_2(x_0)
#         x_out = self.pool(x_out)
#         return x_out 

class projectors(nn.Module):
    def __init__(self, input_nc=4, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        return x_out

class projectors_feature(nn.Module):
    def __init__(self, input_nc=96, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors_feature, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(4, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
        self.conv_0 = nn.Conv2d(input_nc, 4, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input_, input): # input, 
        input_ = self.conv_0(input_) 
        input = input + input_
        # input = input_
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        return x_out
    
class MLP(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.relu = nn.ReLU(inplace = True)  
        self.linear = nn.Linear(in_dims, out_dims) # , padding=1 # 注意，conv不是一个1 * 1的东西，而是一个3 * 3的东西，所以很可能融合了一些不该融合的东西
        self.ln = nn.LayerNorm(out_dims)

    def forward(self, x):
        y_1 = self.linear(x)
        y_1 = self.ln(y_1)
        y_1 = self.relu(y_1)

        return y_1

class projectors_local(nn.Module):
    def __init__(self, input_nc=768, ndf=768, norm_layer=nn.BatchNorm2d):
        super(projectors_local, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
    def forward(self, input):
        input = seq2img(input)
        x_0 = self.conv_1(input)
        # x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        # x_out = self.pool(x_out)
        return x_out
    
class projectors_linear(nn.Module):
    def __init__(self, input_nc=4, ndf=8, norm_layer=nn.LayerNorm):
        super(projectors_linear, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = MLP(input_nc, ndf)
        self.linear_2 = MLP(ndf, ndf*2)
        # self.norm = norm_layer(ndf)
        # self.final = nn.Conv2d(input_nc, 4, kernel_size=1) # , bias=False
        # self.final2 = nn.Conv2d(4, ndf, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input):
        # input = self.final(input) # 居然还不如直接变成class的表现，令人遗憾
        # input = self.final2(input)
        # x_0 = self.conv_1(torch.cat((input_, input), dim = 1))      
        x_0 = self.linear_1(input)
        # x_0 = self.pool(x_0)  
        x_out = self.linear_2(x_0)
        # x_out = self.norm(x_out)
        # x_out = self.pool(x_out)
        return x_out

class projectors_linear_p(nn.Module):
    def __init__(self, input_nc=2, ndf=4, norm_layer=nn.LayerNorm):
        super(projectors_linear_p, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = MLP(input_nc, ndf)
        self.linear_2 = MLP(ndf, ndf*2)
        # self.norm = norm_layer(ndf)
        # self.final = nn.Conv2d(input_nc, 4, kernel_size=1) # , bias=False
        # self.final2 = nn.Conv2d(4, ndf, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input):
        # input = self.final(input) # 居然还不如直接变成class的表现，令人遗憾
        # input = self.final2(input)
        # x_0 = self.conv_1(torch.cat((input_, input), dim = 1))      
        x_0 = self.linear_1(input)
        # x_0 = self.pool(x_0)  
        x_out = self.linear_2(x_0)
        # x_out = self.norm(x_out)
        # x_out = self.pool(x_out)
        return x_out
    
class projectors_linear_p2(nn.Module):
    def __init__(self, input_nc=96, ndf=8, norm_layer=nn.LayerNorm):
        super(projectors_linear_p2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = MLP(input_nc, ndf)
        self.linear_2 = MLP(ndf, ndf*2)
        # self.norm = norm_layer(ndf)
        # self.final = nn.Conv2d(input_nc, 4, kernel_size=1) # , bias=False
        # self.final2 = nn.Conv2d(4, ndf, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input):
        # input = self.final(input) # 居然还不如直接变成class的表现，令人遗憾
        # input = self.final2(input)
        # x_0 = self.conv_1(torch.cat((input_, input), dim = 1))      
        x_0 = self.linear_1(input)
        # x_0 = self.pool(x_0)  
        x_out = self.linear_2(x_0)
        # x_out = self.norm(x_out)
        # x_out = self.pool(x_out)
        return x_out
    
class projectors_linear_unet(nn.Module):
    def __init__(self, input_nc=16, ndf=16, norm_layer=nn.LayerNorm):
        super(projectors_linear_unet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = MLP(input_nc, ndf)
        self.linear_2 = MLP(ndf, ndf)
        # self.norm = norm_layer(ndf)
        # self.final = nn.Conv2d(input_nc, 4, kernel_size=1) # , bias=False
        # self.final2 = nn.Conv2d(4, ndf, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input):
        # input = self.final(input) # 居然还不如直接变成class的表现，令人遗憾
        # input = self.final2(input)
        # x_0 = self.conv_1(torch.cat((input_, input), dim = 1))      
        x_0 = self.linear_1(input)
        # x_0 = self.pool(x_0)  
        x_out = self.linear_2(x_0)
        # x_out = self.norm(x_out)
        # x_out = self.pool(x_out)
        return x_out
    
class projectors_linear_before(nn.Module):
    def __init__(self, input_nc=96, ndf=8, norm_layer=nn.LayerNorm):
        super(projectors_linear_before, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.linear_1 = MLP(input_nc, ndf)
        self.linear_2 = MLP(ndf, ndf*2)
        # self.norm = norm_layer(ndf)
        # self.final = nn.Conv2d(input_nc, 4, kernel_size=1) # , bias=False
        # self.final2 = nn.Conv2d(4, ndf, kernel_size=1)
    # def forward(self, input, input_):
    def forward(self, input):
        # input = self.final(input) # 居然还不如直接变成class的表现，令人遗憾
        # input = self.final2(input)
        # x_0 = self.conv_1(torch.cat((input_, input), dim = 1))      
        x_0 = self.linear_1(input)
        # x_0 = self.pool(x_0)  
        x_out = self.linear_2(x_0)
        # x_out = self.norm(x_out)
        # x_out = self.pool(x_out)
        return x_out
    
class classifier(nn.Module):
    def __init__(self, inp_dim = 4,ndf=8, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.conv_3 = conv(ndf*2, ndf*4)
        self.final = nn.Conv2d(ndf*4, ndf*4, kernel_size=1)
        # self.linear = nn.Linear(in_features=ndf*4*18*12, out_features=1024)
    def forward(self,input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        x_1 = self.pool(x_1)
        x_2 = self.conv_3(x_1)
        x_2 = self.pool(x_2)
        # x_out = self.linear(x_2)
        x_out = self.final(x_2)

        return x_out

class projectors_unet(nn.Module):
    def __init__(self, inp_dim = 4,ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors_unet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        # self.conv_3 = conv(ndf*2, ndf*4)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
        # self.linear = nn.Linear(in_features=ndf*4*18*12, out_features=1024)
    def forward(self,input):
        x_0 = self.conv_1(input)
        # x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        # x_1 = self.pool(x_1)
        # x_2 = self.conv_3(x_1)
        # x_2 = self.pool(x_2)
        # x_out = self.linear(x_2)
        x_out = self.final(x_1)

        return x_out
      
            