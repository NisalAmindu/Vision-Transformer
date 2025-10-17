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

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform_train)


train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#_______________________________________________________________________________




torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)

max_iter = 1000
lr = 6e-4


vit = ViT(Config())
vit = vit.to(device)
vit = torch.compile(vit)


use_fused = True if torch.cuda.is_available() else False
optimizer = torch.optim.AdamW(vit.parameters(),lr = lr,fused = use_fused)

losses = torch.zeros((max_iter,))

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
    optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()
    dt = (t1-t0) * 1000 # ms

    losses[i] = loss.item()

    if i%100 ==0 : print(f'{i}/{max_iter}  {loss.item():.4f}  {dt:.4f} ms')




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