import numpy as np
import pandas as pd
import os
import json
import time
from itertools import chain
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm_notebook
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
from albumentations.augmentations import transforms
from albumentations.pytorch.transforms import ToTensor
from albumentations.core.composition import Compose
from efficientnet_pytorch import EfficientNet
import argparse
from tensorboardX import SummaryWriter

path = '/data/inaturalist2019/'

# load train paths and labels

with open(os.path.join(path, 'train2019.json')) as data_file:
        train_anns = json.load(data_file)
train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
df_train_file = pd.merge(train_img_df, train_anns_df, on='image_id')
del train_anns
del train_anns_df
del train_img_df

# load test paths

with open(os.path.join(path, 'test2019.json')) as data_file:
        test_anns = json.load(data_file)
df_test_file = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
del test_anns

df_train, df_valid = train_test_split(df_train_file, stratify=df_train_file['category_id'], test_size=0.2, random_state=8)

# parameters

par = dict(
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    model_name = 'efficientnet-b2',
    n_train = 265213,
    n_test = 35350,
    num_classes = 1010,
    batch_size = 128,
    n_workers = 12
    )
par['n_train_train'] = df_train.shape[0]
par['n_train_valid'] = df_valid.shape[0]
par['n_steps'] = int(np.ceil(par['n_train_train']/par['batch_size']))
par['image_size'] = EfficientNet.get_image_size(par['model_name']) 

# utils

def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the 5 top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res
    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path,filename))

# define dataset

class MyDataset(Dataset):
    def __init__(self, split_data, data_root=path, transform=None, train=True):
        super().__init__()
        self.df = split_data.iloc[:,1:].values
        self.data_root = data_root
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if self.train:
            img_name, label = self.df[index]
            img_path = os.path.join(self.data_root, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, label
        else:
            img_name = self.df[index]
            img_path = os.path.join(self.data_root, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image
        
# transforms

train_transforms = Compose([
    transforms.SmallestMaxSize(par['image_size']),
    transforms.RandomCrop(par['image_size'], par['image_size']),
    transforms.RandomRotate90(),
    transforms.Flip(),
#     transforms.RandomGamma(),
    transforms.Normalize()
])

valid_transforms = Compose([
    transforms.SmallestMaxSize(par['image_size']),
    transforms.RandomCrop(par['image_size'], par['image_size']),
    transforms.Normalize()
])

plot_transforms = Compose([
    transforms.SmallestMaxSize(par['image_size']),
    transforms.RandomCrop(par['image_size'], par['image_size']),
    transforms.RandomRotate90(),
    transforms.Flip(),
#     transforms.RandomGamma()
#     ,transforms.Normalize()
])

train_data = MyDataset(split_data=df_train, transform=train_transforms)
valid_data = MyDataset(split_data=df_valid, transform=valid_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=par['batch_size'], shuffle=True, num_workers=par['n_workers'])
valid_loader = DataLoader(dataset=valid_data, batch_size=par['batch_size'], shuffle=False, num_workers=par['n_workers'])


train_writer = SummaryWriter(os.path.join(path, 'train_logs'))
valid_writer = SummaryWriter(os.path.join(path, 'val_logs'))

def train_epoch(model, optimizer, criterion, gl_iter):
    losseslog = []
    top1log = []
    top5log = []
    
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):

        data = x_batch.to(par['device'], dtype=torch.float)
        data = torch.transpose(data, 1,3)
        target = y_batch.to(par['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = torch.max(output, 1)[1]
                
        acc1, acc5 = accuracy(output.data, target.data)
#         losseslog.append(loss.item())
#         top1log.append(acc1.item())
#         top5log.append(acc5.item())
        
        train_writer.add_scalar('loss', loss.item(), global_step=gl_iter)
        train_writer.add_scalar('acc@1', acc1.item(), global_step=gl_iter)
        train_writer.add_scalar('acc@5', acc5.item(), global_step=gl_iter)
        gl_iter += 1 
        
        loss.backward()
        optimizer.step()
        
    return gl_iter#, losseslog, top1log, top5log 
        
def valid_epoch(model, criterion, gl_iter):
    losseslog = []
    top1log = []
    top5log = []
    
    model.eval()
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        data = x_batch.to(par['device'], dtype=torch.float)
        data = torch.transpose(data, 1,3)
        target = y_batch.to(par['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = torch.max(output, 1)[1]
        
        acc1, acc5 = accuracy(output.data, target.data)
#         losseslog.append(loss.item())
#         top1log.append(acc1.item())
#         top5log.append(acc5.item())
        
        valid_writer.add_scalar('loss', loss.item(), global_step=gl_iter)
        valid_writer.add_scalar('acc@1', acc1.item(), global_step=gl_iter)
        valid_writer.add_scalar('acc@5', acc5.item(), global_step=gl_iter)
        gl_iter += 1
        
        loss.backward()
        optimizer.step()
                
    return gl_iter#, np.mean(losseslog), np.mean(top1log), np.mean(top5log)

def train(model, optimizer, criterion, batchsize, n_epochs):
    
    global_iter = 0
    for epoch in range(n_epochs):
        
        print('train epoch ',epoch)
        global_iter = train_epoch(model, optimizer, criterion, global_iter)

        print('valid epoch ',epoch)
        global_iter = valid_epoch(model, criterion, global_iter)

        scheduler.step()
        
# Classify with EfficientNet

model = EfficientNet.from_pretrained(par['model_name'])
for params in model.parameters():
    params.requires_grad = False
n_fc_in_features = model._fc.in_features
model._fc = nn.Linear(n_fc_in_features, par['num_classes'])
params_to_train = chain(model._conv_head.parameters(), model._bn1.parameters(), model._fc.parameters())
model = model.to(par['device'])

state1 = torch.load(os.path.join(path,'checkpoint2.pth.tar'))
model.load_state_dict(state1['state_dict'])

for params in model._blocks[-1]._expand_conv.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._bn0.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._depthwise_conv.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._bn1.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._se_reduce.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._se_expand.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._project_conv.parameters():
    params.requires_grad = True
for params in model._blocks[-1]._bn2.parameters():
    params.requires_grad = True

new_params_to_train = chain(model._conv_head.parameters(), model._bn1.parameters(), model._fc.parameters(),
                            model._blocks[-1]._expand_conv.parameters(),
                            model._blocks[-1]._bn0.parameters(),
                            model._blocks[-1]._depthwise_conv.parameters(),
                            model._blocks[-1]._bn1.parameters(),
                            model._blocks[-1]._se_reduce.parameters(),
                            model._blocks[-1]._se_expand.parameters(),
                            model._blocks[-1]._project_conv.parameters(),
                            model._blocks[-1]._bn2.parameters())


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params_to_train, 1e-3)
scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=1000)

train(model, optimizer, criterion, par['batch_size'], 10)

state0 = ({'state_dict': model.state_dict(),
           'optimizer': optimizer.state_dict()})
save_checkpoint(state0, 'checkpoint777.pth.tar')  