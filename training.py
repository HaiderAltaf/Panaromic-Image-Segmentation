#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import UNET
import tqdm
from dataset_ import SegmentationDataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import cv2


# In[3]:


img_dir="/dataset/train/rgb/"
mask_dir="/dataset/train/segmentation_color/"
train_transform=A.Compose([
    A.Resize(height=100,width=720),
    A.Rotate(limit=35,p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0,0.0,0.0],
        std=[1.0,1.0,1.0],
        max_pixel_value=255
    ),
    ToTensorV2(),
])

train_data = SegmentationDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )
train_loader = DataLoader(
        train_data,
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )

val_transform=A.Compose([
    A.Resize(height=100,width=720),
    A.Normalize(
    mean=[0.0,0.0,0.0],
    std=[1.0,1.0,1.0],
    max_pixel_value=255
    ),
    ToTensorV2(),
    
])
eval_data=SegmentationDataset(image_dir=img_dir,
        mask_dir=mask_dir,
        transform=val_transform)
eval_loader=DataLoader(eval_data,batch_size=1,num_workers=0)


# In[3]:


device=("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


model=UNET().to(device)
status=False
lossfun=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
epochs=100
if status==True:
    pass
    #model.load_state_dic
for ephs in range(epochs):
    checkpoint={
        "state_dict":model.state_dict(),
        "optimizer:":optimizer.state_dict()
    }
    torch.save(checkpoint,"my_checkpoint.pth.tar")
    for img,mask in train_loader:
        train_img=img.to(device)
        train_img=train_img/train_img.max()
        mask_img=mask.to(device)
        mask_img=mask_img.permute(0,3,1,2)
        mask_img=mask_img/mask_img.max()
        y_pred=model(train_img)
        loss=lossfun(y_pred,mask_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch{ephs+1} completed with a loss of {loss.cpu().detach()}")


# In[37]:


flag=0
for img,mask in eval_loader:
        train_img=img.to(device)
        train_img=train_img/train_img.max()
        mask_img=mask.to(device)
        mask_img=mask_img.permute(0,3,1,2)
        mask_img=mask_img/mask_img.max()
        y_pred=model(train_img)
        maskimg=np.array(mask_img.detach())
        maskimg=np.squeeze(maskimg).transpose(1,2,0)
        predimg=np.array(y_pred.detach())
        predimg=np.squeeze(predimg).transpose(1,2,0)
        predimg=predimg/predimg.max()
        saveimg=cv2.hconcat([maskimg,predimg])
        saveimg=cv2.cvtColor(saveimg,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',saveimg[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(f'{flag}.png',predimg)
        flag+=1


# In[34]:


plt.figure(0,figsize=(15,15))
img1=np.squeeze(np.array(y_pred.detach()))
cv2.imshow('frame',saveimg[:,:,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


predimg.max()


# In[21]:


img1.shape


# In[ ]:




