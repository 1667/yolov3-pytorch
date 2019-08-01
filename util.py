#coding=utf-8
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes,CUDA = True):

    """
    在特征图上进行多尺度预测, 在GRID每个位置都有三个不同尺度的锚点.predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
    是在网络输入图片坐标系下的值,s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量) 
    维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式。参数见predict_transform()里面的变量。
    并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
    """

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2) # 下采样倍数，也就是缩放了多少
    grid_size = inp_dim // stride # 当前的图像大小
    bbox_attrs = 5+num_classes
    num_anchors = len(anchors)
    # 输入的尺寸是 grid_size*grid_size*num_anchors*（4+1+num_classes）比如：13*13*3*（4+1+80）
    # 4 是边框坐标 1 是边框置信度 3 是先验框的个数
    if CUDA:
        prediction = prediction.cuda()

    # -----------------------------------
    # 维度调整
    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size) # torch.Size([1, 255, 169])
    # print("pre1",prediction.size())
    prediction = prediction.transpose(1,2).contiguous()# torch.Size([1, 169, 255])
    # print("pre2",prediction.size())
    # 将 anchor 按行排列，即一行对应一个anchor属性，
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs) # torch.Size([1, 507, 85])
    # print("pre3",prediction.size())
    # -----------------------------------

    anchors = [(a[0]/stride,a[1]/stride) for a in anchors] # 先验框也缩放到对应大小

    # print(prediction[:,:,4])
    # -----------------------------------
    # 对 centerx centery 和置信度取sigmoid
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    # -----------------------------------

    # -----------------------------------
    #调整中心坐标 分别加上对应网格在整个图像的起始坐标，如 第一个是（0，0） 最后一个是（12，12）
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
    # print("xy center ",prediction[:,:,:2])
    # -----------------------------------

    # -----------------------------------
    # 求先验框在当前特征图上的宽高
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    # print("anchors ",anchors)
    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    # print("anchors size2 ",anchors.size())
    # 根据公式bw=pw×e^tw及bh=ph×e^th，求边框在当前特征图的尺寸
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    # print("anchors ===",prediction[:,:,2:4])
    # -----------------------------------

    # -----------------------------------
    # 求每个分类的得分
    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])

    # 调整预测边框的大小和目标边框尺寸一致，为了和目标边框大小作比较，求代价函数
    # 最终把所有的坐标映射到输入的图片上
    prediction[:,:,:4] *= stride

    return prediction
    







