import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

from datasets import load_dataset
from dataclasses import dataclass
from PIL import Image
import time
import math
import os

from vit import ViT


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


# Data pre ----------------------------------------------------------------------------

batch_size = 32

# Load dataset (dataset object : "train" : 126689 images , "val" : 5000 imagess , 100 classes) 
dataset = load_dataset("/home/samitha/Projects/datasets/imagenet100")


# Image Transformation with Data Augmentation
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3 channels
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),   # ViT style
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3 channels
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])





class ImageNet100(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image']
        label = sample['label']
        if self.transform:
            img = self.transform(img)
        return img, label
    
# wrappers
train_ds = ImageNet100(dataset['train'],train_transform)
val_ds = ImageNet100(dataset['validation'],val_transform)


# Efficient Dataloaders
train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True)
val_loader = DataLoader(val_ds,batch_size=batch_size,num_workers=8,pin_memory=True,persistent_workers=True)



# LR Schedule ----------------------------------------------------------------------

# epochs = 100
max_iter = 400000 # epochs * 3959
warmup_steps = max_iter * 0.05
max_lr = 3e-4
min_lr = max_lr * 0.1


def get_lr(i):
    
    # warmup stage : linear
    if i < warmup_steps :
        return (max_lr/warmup_steps) * (i+1)

    if i > max_iter:
        return min_lr

    # cosine dacay
    decay_ratio = (i-warmup_steps) / (max_iter-warmup_steps)
    assert 0<= decay_ratio <=1
    c = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + c * (max_lr - min_lr)    



# Model ----------------------------------------------------------------------------

@dataclass
class ViTBaseConfig:
    num_classes: int = 100        # Imagenet 100
    img_size: int = 224      
    im_channels: int = 3
    patch_size: int = 16

    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 768

    dropout = 0.1  

    @property
    def n_patch(self):
        return (self.img_size//self.patch_size)**2
    

# Model
vit = ViT(ViTBaseConfig())
print(f'{sum([p.numel() for p in vit.parameters()])/10**6} M')

vit = vit.to(device)               # ship model to GPU VRAM
vit_compile = torch.compile(vit)   # wapper but use VRAM vit params



# Dir for save checkpoints ----------------------------------------------------------
 
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

with open(log_file, "w") as f: 
    pass 


# Train ----------------------------------------------------------------------------


use_fused = torch.cuda.is_available()
optimizer = torch.optim.AdamW(vit_compile.parameters(),lr = max_lr,weight_decay=0.1,fused=use_fused)


train_iter = iter(train_loader)
# step = 0
# for i in range(epochs):
    
for step in range(max_iter):     # 3959.03125 baches for 1 train epoch
    
    t0 = time.time()
    
    # Val -----------------------------------------------------------------------
    if step==0 or step%1000==0 or step == max_iter:
        
        val_step = 0
        val_loss = 0.0
        vit_compile.eval()
        with torch.no_grad():
            for x,y in val_loader:    # 156.25 for 1 val epoch
                x,y = x.to(device),y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits , loss = vit_compile(x,y)
                
                val_loss += loss.detach()
                val_step+=1
            val_loss /= val_step
        print(f'Validation loss : {val_loss.item():.4f}')
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss.item():.4f}\n")
        vit_compile.train()
    
    
    
    # Checkpoints ---------------------------------------------------------------
    if step>0 and (step%50000==0 or step==max_iter-1):
        
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': vit.state_dict(),
            'config': vit.config,
            'step': step,
            'val_loss': val_loss.item(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        
    
    
    
    # Train ---------------------------------------------------------------------
    try:
        xb,yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb,yb = next(train_iter)
        
    xb,yb = xb.to(device),yb.to(device)
    
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits , loss = vit_compile(xb,yb)
        
    loss.backward()
    
    # Gradient Clipping
    norm = torch.nn.utils.clip_grad_norm_(vit_compile.parameters(), 1.0)
    
    # LR Schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr  # update optimizer
        
    optimizer.step()
    # ---------------------------------------------------------------------------
    
    t1 = time.time()
    dt = (t1 - t0)*1000  # ms
    
    print(f'{step}/{max_iter}  {loss.item():.4f}  {dt:.4f} ms  norm:{norm.item():.4f}  lr:{lr:.4e}')
    
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss.item():.6f}\n")
    
    # step+=1