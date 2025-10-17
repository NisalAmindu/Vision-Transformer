import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dataclasses import dataclass
import time
import math

from vit import ViT,Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'




#_______________________________________________________________________________

#Get data

batch_size = 32

transform_train = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_train)


train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#_______________________________________________________________________________




torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)




vit = ViT(Config(img_size=32,im_channels=3,patch_size=8,n_head=8,n_layer=8,n_embd=1024))
vit = vit.to(device)
vit = torch.compile(vit)


#------------------------------------------------------------------------------4

max_iter = 10000
warm_up = max_iter *0.05
max_lr = 6e-4
min_lr = max_lr * 0.1



def get_lr(i):

    if i < warm_up:
        return (max_lr/warm_up) * (i+1)

    if i>max_iter :
        return min_lr

    # cosine decay
    diff = max_lr - min_lr
    steps = max_iter - warm_up
    lr = (diff/2) * math.cos(i * (math.pi / steps)) + diff
    return lr


#------------------------------------------------------------------------------


losses = torch.zeros((max_iter,))
lrs = torch.zeros((max_iter,))
norms = torch.zeros((max_iter,))



# Optimizer
use_fused = True if torch.cuda.is_available() else False
optimizer = torch.optim.AdamW(vit.parameters(),lr = max_lr,weight_decay = 0.1,fused = use_fused)


train_data_iter = iter(train_data)

for i in range(max_iter):


    t0 = time.time()

    try:
        xb,yb = next(train_data_iter)
    except StopIteration:
         train_data_iter = iter(train_data)   # reset
         xb,yb = next(train_data_iter)

    xb,yb = xb.to(device),yb.to(device)


    logits , loss = vit(xb,yb)

    optimizer.zero_grad()
    loss.backward()


    # LR Schedule
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Gradient Clipping
    norm = torch.nn.utils.clip_grad_norm_(vit.parameters(),1.0)

    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()
    dt = (t1-t0) * 1000 # ms

    losses[i] = loss.item()
    lrs[i] = lr
    norms[i] = norm

    if i%100 ==0 : print(f'{i}/{max_iter}  {loss.item():.4f}  {dt:.4f} ms   norm:{norm.item():.4f}   lr:{lr:.4e}')




#_______________________________________________________________________________



# Validation accuracy


correct, total = 0, 0
for xb, yb in val_data:
    xb,yb = xb.to(device),yb.to(device)
    logits = vit(xb)
    preds = torch.argmax(logits, dim=-1)
    correct += (preds == yb).sum().item()
    total += yb.size(0)

val_acc = correct / total
print(f"Validation accuracy: {val_acc:.4f}")