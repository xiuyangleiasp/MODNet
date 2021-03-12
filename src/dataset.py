# -*- coding: utf-8 -*-
"""
Created on Fri May 04 15:07:48 2018

@author: lxy
"""

import torch
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import cv2
import src.transforms as transforms
from scipy.ndimage import grey_dilation,grey_erosion

class HumanMattingDataLoader(object):
    
    def __init__(self,files,color_channel="RGB",resize=512,
                 padding_value=0,crop_range=[0.75,1],flip_hor=0.5,rotate=.03,angle=10,noise_std=5,
                normalize=True,is_training=True,shuffle=True,batch_size=1,n_workers=1,pin_memory=True):
       
        super(HumanMattingDataLoader,self).__init__()
        
        self.files=files
        self.color_channel=color_channel
        self.resize=resize
        self.rotate=rotate
        self.flip_hor=flip_hor
        self.normalize=normalize
        self.is_training=is_training
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.n_workers=n_workers
        self.pin_memory=pin_memory
        self.padding_value=padding_value
        self.crop_range=crop_range
        self.angle=angle
        self.noise_std=noise_std

        self.dataset=HumanMattingDataset(self.files,
                                         self.color_channel,
                                         self.resize,
                                         self.padding_value,
                                         self.is_training,
                                         self.crop_range,
                                         self.flip_hor,
                                         self.rotate,
                                         self.angle,
                                         self.noise_std,
                                         self.normalize
                                         )

        @property
        def loader(self):
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory)


class HumanMattingDataset(Dataset):
    
    def __init__(self,files,color_channel="RGB",resize=512,padding_value=0,
                  is_training=True,crop_range=[0.75,1.0],
                  flip_hor=0.5,rotate=0.3,angle=10,noise_std=5,
                  normalize=True,
                  mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]):

        self.image_files,self.label_files=[],[]
        self.trimap_files=[]
        
        fp=open(files,"r")

        lines=fp.read().split("\n")
        lines=[line.strip() for line in lines if len(line)]
        fp.close()

        for line in lines:
            error_flg=False
            image_file=line
            label_file=image_file
            label_file=label_file.replace("JPEGImages","labels")
            label_file=label_file.replace(".jpg",".png")
            trimap_file=label_file.replace("labels","trimap")
            if not os.path.exists(image_file):
                image_file=image_file.replace(".jpg",".png")
                if not os.path.exists(image_file):
                    print("%s does not exist!"%(image_file))
                    error_flg=True
            if not os.path.exists(label_file):
                label_file=label_file.replace(".png",".jpg")
                if not os.path.exists(label_file):
                    print("%s does not exit!"%(label_file))
                    error_flg=True
            if error_flg==False:
                self.image_files.append(image_file)
                self.label_files.append(label_file)
                self.trimap_files.append(trimap_file)
        
        self.color_channel=color_channel
        self.resize=resize
        self.padding_value=padding_value
        self.is_training=is_training
        self.crop_range=crop_range
        self.flip_hor=flip_hor
        self.rotate=rotate
        self.angle=angle
        self.noise_std=noise_std
        self.normalize=normalize
        self.mean=np.array(mean)[None,None,:]
        self.std=np.array(std)[None,None,:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self,idx):

        img_file=self.image_files[idx]
        label_file=self.label_files[idx]
        #trimap_file=self.trimap_files[idx]

        image=cv2.imread(img_file)
        label=cv2.imread(label_file,0)
        #trimap=cv2.imread(trimap_file,0)
        trimap=None
        if self.is_training:
            #image=transforms.random_noise(image,std=self.noise_std)
            image,label,trimap=transforms.flip_horizon(image,label,trimap,self.flip_hor)
            #image,label,trimap=transforms.rotate_90(image,label,trimap,self.rotate)
            #image,label,trimap=transforms.rotate_angle(image,label,trimap,self.angle)
            #image,label,trimap=transforms.random_crop(image,label,trimap,self.crop_range)
   
        image=transforms.resize_image(image,expected_size=self.resize,pad_value=self.padding_value,mode=cv2.INTER_LINEAR)
        label=transforms.resize_image(label,expected_size=self.resize,pad_value=self.padding_value,mode=cv2.INTER_LINEAR)
        if trimap:
            trimap=transforms.resize_image(trimap,expected_size=self.resize,pad_value=self.padding_value,mode=cv2.INTER_LINEAR)
        
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.normalize:
            image=image.astype(np.float32)/255.0
            image=(image-self.mean)/self.std
        image=np.transpose(image,axes=(2,0,1))
        
        #label[label>10] = 1
        label=label.astype(np.float32)/255.0
        label[label<0.04]=0
        
        if trimap:
            trimap=trimap.astype(np.float32)/255.0
        else:
            trimap=(label>=0.9).astype(np.float32)
            not_bg=(label>0).astype(np.float32)
            d_size=self.resize//256 * 15
            e_size=self.resize//256 * 15
            trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        
        image=torch.tensor(image.copy(),dtype=torch.float32)
        trimap=torch.tensor(trimap.copy(),dtype=torch.float32)
        label=torch.tensor(label.copy(),dtype=torch.float32)
        trimap=torch.unsqueeze(trimap,0)
        label=torch.unsqueeze(label,0)

        return image,trimap,label
