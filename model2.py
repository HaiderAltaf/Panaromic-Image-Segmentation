#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules import Conv2d
class Doubleconv(nn.Module):
  def __init__(self,inchannel,outchannel):
    super().__init__()
    self.conv_layer=nn.Sequential(
        nn.Conv2d(inchannel,outchannel,kernel_size=(3,3),stride=1,padding=1),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(),
        nn.Conv2d(outchannel,outchannel,kernel_size=(3,3),stride=1,padding=1),
        nn.BatchNorm2d(outchannel),
        nn.ReLU(),
    )
  def forward(self,x):
    return self.conv_layer(x)

class UNET2(nn.Module):
  def __init__(self,feature=[64,128,256,512]):
    super().__init__()

    self.encoder1=Doubleconv(3,feature[0])
    self.encoder2=Doubleconv(feature[0],feature[1])
    self.encoder3=Doubleconv(feature[1],feature[2])
    self.encoder4=Doubleconv(feature[2],feature[3])

    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

    feature1=feature[::-1]
    self.middle=Doubleconv(512,1024)
    self.transpose1=nn.ConvTranspose2d(feature1[0]*2,feature1[0],kernel_size=(2,2),stride=2)
    self.decode1=Doubleconv(feature1[0]*2,feature1[0])
    self.transpose2=nn.ConvTranspose2d(feature1[0],feature1[1],kernel_size=(2,2),stride=2)
    self.decode2=Doubleconv(feature1[0],feature1[1])
    self.transpose3=nn.ConvTranspose2d(feature1[1],feature1[2],kernel_size=(2,2),stride=2)
    self.decode3=Doubleconv(feature1[1],feature1[2])
    self.transpose4=nn.ConvTranspose2d(feature1[2],feature1[3],kernel_size=(2,2),stride=2)
    self.decode4=Doubleconv(feature1[2],feature1[3])
    self.final=nn.ConvTranspose2d(feature1[3],1,kernel_size=(1,1))
 

  def forward(self,x):
    
    self.down=[]
    y=self.encoder1(x)
    self.down.append(y)
    y=self.pool(y)
    
    y=self.encoder2(y)
    self.down.append(y)
    y=self.pool(y)

    y=self.encoder3(y)
    self.down.append(y)
    y=self.pool(y)

    y=self.encoder4(y)
    self.down.append(y)
    y=self.pool(y)

    y=self.middle(y)

    y=self.transpose1(y)
    if y.shape!=self.down[-1]:
      y=TF.resize(y,size=self.down[-1].shape[2:])
    concat=torch.cat((self.down[-1],y),axis=1)
    y=self.decode1(concat)

    y=self.transpose2(y)
    if y.shape!=self.down[-2]:
      y=TF.resize(y,size=self.down[-2].shape[2:])
    concat=torch.cat((self.down[-2],y),axis=1)
    y=self.decode2(concat)

    y=self.transpose3(y)
    if y.shape!=self.down[-3]:
      y=TF.resize(y,size=self.down[-3].shape[2:])
    concat=torch.cat((self.down[-3],y),axis=1)
    y=self.decode3(concat)

    y=self.transpose4(y)
    if y.shape!=self.down[-4]:
      y=TF.resize(y,size=self.down[-4].shape[2:])
    concat=torch.cat((self.down[-4],y),axis=1)
    y=self.decode4(concat)
    y=self.final(y)

    return y


# In[ ]:




