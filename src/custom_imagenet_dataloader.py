import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data pre ----------------------------------------------------------------------------

batch_size = 32

# Load dataset (dataset object : "train" : 126689 images , "val" : 5000 imagess , 100 classes) 
dataset = load_dataset("/home/samitha/Projects/datasets/imagenet100")


# Image Transformation with Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),   # ViT style
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class DataLoader():
    def __init__(self,data,transform,batch_size,Shuffle= False):
        
        self.transform = transform
        self.batch_size = batch_size
        
        self.data = list(zip(data['image'],data['label']))  # make (image,label) tuples list
        if Shuffle:
            random.shuffle(self.data)    # in place random shuffle of data 
            
        self.count = 0
        
        
    def get_batch(self):
        
        batch = self.data[self.count : self.count + self.batch_size]  # taking batch
        imgs,labels = zip(*batch)    # (img1,lb1),(img2,lb2) -> (im1,img2) , (lb2,lb2)
        
        xb = torch.stack([self.transform(img) for img in imgs])
        yb = torch.tensor(labels)
        
        self.count += self.batch_size   # update for next batch
        if self.count > len(self.data): # Reset for next epoch
            self.count = 0 
            random.shuffle(self.samples)
            
        return xb,yb 
        
        
        
# Initialize Data train,val
train_loader = DataLoader(dataset['train'],train_transform,batch_size,Shuffle=True)
val_loader = DataLoader(dataset['validation'],val_transform,batch_size)


# temp get data
xb,yb = train_loader.get_batch()
xb,yb = xb.to(device),yb.to(device)
print(xb,yb)



# Train ----------------------------------------------------------------------------