# -*- coding: utf-8 -*-
"""
Created on Fri May 04 15:07:48 2018

@author: lxy
"""

import cv2
import numpy as np

def random_crop(image,label,trimap,crop_range):

    if crop_range[0]==crop_range[1] and crop_range[0]==1.0:
        return image,label
    
    crop_ratio=np.random.choice(np.linspace(crop_range[0],crop_range[1],num=10),size=())

    H,W=label.shape
    size=H if H<W else W
    size=int(size*crop_ratio)
    max_i,max_j=H-size,W-size

    i=np.random.choice(np.arange(0,max_i+1),size=())
    j=np.random.choice(np.arange(0,max_j+1),size=())

    image_cropped=image[i:i+size,j:j+size,:]
    label_cropped=label[i:i+size,j:j+size]
    if trimap:
        trimap_cropped=trimap[i:i+size,j:j+size]
    else:
        trimap_cropped=None
    return image_cropped,label_cropped,trimap_cropped

def flip_horizon(image,label,trimap,prob):

    if prob:
        if np.random.choice([False,True],size=(),p=[1-prob,prob]):
            image=np.flip(image,axis=1)
            label=np.flip(label,axis=1)
            if trimap:
                trimap=np.flip(trimap,axis=1)
    return image,label,trimap


def rotate_90(image,label,trimap,prob):
    
    if prob:
        k=np.random.choice([-1,0,1],size=(),p=[prob/2,1-prob,prob/2])
        if k:
            image=np.rot90(image,k=k,axes=(0,1))
            label=np.rot90(label,k=k,axes=(0,1))
            if trimap:
                trimap=np.rot90(trimap,k=k,axes=(0,1))
    return image,label,trimap

def rotate_angle(image,label,trimap,angle_max):

    if angle_max:
        angle=np.random.choice(np.linspace(-angle_max,angle_max,num=21),size=())
        (h,w)=image.shape[:2]
        (cX,cY)=(w//2,h//2)

        M=cv2.getRotationMatrix2D((cX,cY),float(angle),1.0)
        cos=np.abs(M[0,0])
        sin=np.abs(M[0,1])
        nw=int((h*sin)+(w*cos))
        nh=int((h*cos)+(w*sin))
        M[0,2]+=(nw/2)-cX
        M[1,2]+=(nh/2)-cY

        image=cv2.warpAffine(image,M,(nw,nh))
        label=cv2.warpAffine(label,M,(nw,nh))
        if trimap:
            trimap=cv2.warpAffine(trimap,M,(nw,nh))
    return image,label,trimap


def random_noise(image,std):

    if std:
        noise=np.random.normal(0,std,size=image.shape)
        image=image+noise
        image[image<0]=0
        image[image>255]=255
        image=image.astype(np.uint8)
    return image

def resize_image(image,expected_size,pad_value,ret_params=False,mode=cv2.INTER_LINEAR):

    '''
    h,w=image.shape[:2]

    if w>h:
        w_new=int(expected_size)
        h_new=int(h*w_new/w)

        image=cv2.resize(image,(w_new,h_new),interpolation=mode)
        
        pad_up=(w_new-h_new)//2
        pad_down=w_new-h_new-pad_up
        if len(image.shape)==3:
            pad_width=((pad_up,pad_down),(0,0),(0,0))
            constant_values=((pad_value,pad_value),(0,0),(0,0))
        elif len(image.shape)==2:
            pad_width=((pad_up,pad_down),(0,0))
            constant_values=((pad_value,pad_value),(0,0))
        
        image=np.pad(image,
                     pad_width=pad_width,
                     mode="constant",
                     constant_values=constant_values)
        if ret_params:
            return image,pad_up,0,h_new,w_new
        else:
            return image
    elif w<h:
        h_new=int(expected_size)
        w_new=int(w*h_new/h)
        image=cv2.resize(image,(w_new,h_new),interpolation=mode)

        pad_left=(h_new-w_new)//2
        pad_right=h_new-w_new-pad_left

        if len(image.shape)==3:
            pad_width=((0,0),(pad_left,pad_right),(0,0))
            constant_values=((0,0),(pad_value,pad_value),(0,0))
        elif len(image.shape)==2:
            pad_width=((0,0),(pad_left,pad_right))
            constant_values=((0,0),(pad_value,pad_value))
        if ret_params:
            return image,0,pad_left,h_new,w_new
        else:
            return image
    else:
        image=cv2.resize(image,(expected_size,expected_size),interpolation=mode)
        if ret_params:
            return image,0,0,expected_size,expected_size
        else:
            return image
    '''
    image=cv2.resize(image,(expected_size,expected_size),interpolation=mode)
    if ret_params:
        return image,0,0,expected_size,expected_size
    else:
        return image
