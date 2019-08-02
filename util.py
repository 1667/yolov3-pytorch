#coding=utf-8
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


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

def bbox_iou(box1,box2):

    b1_x1,b1_y1,b1_x2,b1_y2 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    b2_x1,b2_y1,b2_x2,b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.max(b1_x2,b2_x2)
    inter_rect_y2 = torch.max(b1_y2,b2_y2)

    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min = 0)*torch.clamp(inter_rect_y2-inter_rect_y1+1,min=0)

    b1_area = (b1_x2 - b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area = (b2_x2 - b2_x1+1)*(b2_y2-b2_y1+1)

    iou = inter_area/(b1_area+b2_area-inter_area)

    return iou
    
def write_results(prediction,confidence,num_classes,nms_conf = 0.4):

    """
    NMS （非极大值抑制过程）
    1.先把置信度比较低的过滤掉
    2.改变维度，使输出只包含类别中最高得分及类别
    3.取出图片中所有的类别，并按类别进行遍历
    4.对置信度排序，并计算iou
    5.去除重合率较高的预测，重复计算，直到对该类别的所有预测遍历完成
    """

    # 将置信度小于阀值的边界框设为零
    conf_mask = (prediction[:,:,4] > confidence).float()
    # print("conf mask1",conf_mask,conf_mask.size())
    # unsqueeze就是 拓展维度
    conf_mask = conf_mask.unsqueeze(2)
    # print("conf mask2",conf_mask,conf_mask.size())
    prediction = prediction*conf_mask

    # 由（center，w,h)变为 （left,top,right,bottom) 
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    # 遍历batch中的每个图片的检测结果
    for ind in range(batch_size):
        # 每个图片都有10627个结果
        image_pred = prediction[ind]

        # 取最大预测值，并把85保存为7（增加了分数和index，去掉了其他项）
        # 最大值 和最大值索引
        max_conf,max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:,:5],max_conf,max_conf_score)
        image_pred = torch.cat(seq,1)
        # print("imagepred size1 ",image_pred.size())
        # 去掉置信度被设为零的行,其实是取出不为零的行
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        # print(non_zero_ind.size())
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        # print("imagepred_ size2 ",image_pred_)
        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:,-1]) # 取出 类型ID 
        # print("img class",img_classes)
        for cls in img_classes:

            #取出属于这个类别的所有预测
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            # 按置信度进行排序，而不是得分
            conf_sort_index = torch.sort(image_pred_class[:,4],descending = True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            # print("pred class",cls,image_pred_class)
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # 去掉重合率较高的预测，所以对于有多个物体的类别，只要没有重合，就不会被过滤。
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                # print("iou ",cls,i,iou_mask)
                image_pred_class[i+1:] *= iou_mask
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                # print("iou non z",non_zero_ind)
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            # 把预测结果和图片在batch中的id对应起来
            # new(image_pred_class.size(0),1)，这个是为了处理同一个类别有多个物体的情况，要把所有预测都和图片对应
            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
            # print("batch ind",batch_ind)
            seq = batch_ind,image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True

            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
            # print(output)

    try:
        return output
    except:
        return 0

def load_classes(namefile):
    fp = open(namefile,"r")
    names = fp.read().split("\n")[:-1]
    return names

# 按比例缩放图片大小，并用128填充满图片
def letterbox_image(img,inp_dim):

    img_w,img_h = img.shape[1],img.shape[0]
    w,h = inp_dim
    new_w = int(img_w*min(w/img_w,h/img_h))
    new_h = int(img_h*min(w/img_w,h/img_h))

    resized_image = cv2.resize(img,(new_w,new_h),interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1],inp_dim[0],3),128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2+new_w,:] = resized_image

    return canvas

def prep_image(img,inp_dim):

    img = letterbox_image(img,(inp_dim,inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img





    








