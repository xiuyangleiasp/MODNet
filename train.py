# -*- coding: utf-8 -*-

import torch
from src.dataset import HumanMattingDataset
from torch.utils.data import DataLoader

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter

bs = 16         # batch size
lr = 0.01       # learn rate
epochs = 40     # total epochs

modnet = torch.nn.DataParallel(MODNet()).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

train_file="train data list"

train_dataset=HumanMattingDataset(files=train_file)
dataloader=DataLoader(train_dataset,batch_size=bs,shuffle=True)
#dataloader = HumanMattingDataLoader(files=train_file)
iter_time=len(dataloader)
for epoch in range(0, epochs):
    for idx, (image, trimap, gt_matte) in enumerate(dataloader):
        image=image.to(device='cuda')
        trimap=trimap.to(device='cuda')
        gt_matte=gt_matte.to(device='cuda')
        semantic_loss, detail_loss, matte_loss = \
            supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
        lr_scheduler.step()
        cur_lr=lr_scheduler.get_lr()
        print(f"epoch: {epoch+1}/{epochs} lr:{cur_lr} iter: {idx}/{iter_time} lr: {lr} semantic_loss: {semantic_loss}, detail_loss: {detail_loss}, matte_loss: {matte_loss}")
    ckpt_name="./checkpoints/MODNet_"+str(epoch + 1)+".ckpt"
    torch.save(modnet,ckpt_name)