import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import os


torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)
    

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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


def get_dataloaders(dataset_path,batch_size = 32):
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Get transforms
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    # Create datasets
    train_ds = ImageNet100(dataset['train'], train_transform)
    val_ds = ImageNet100(dataset['validation'], val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers= 8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers= 8 ,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader




def get_val_dataloader(dataset_path,batch_size = 32):
    
    data_dir = os.path.join(dataset_path, "data")
    val_file = os.path.join(data_dir, "validation-00000-of-00001.parquet")
    
    val_dataset = load_dataset(
        'parquet',
        data_files=val_file,
        split='train'  
    )
    
    val_transform = get_transforms(is_train=False)
    
    val_ds = ImageNet100(val_dataset, val_transform)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers= 8 ,
        pin_memory=True,
        persistent_workers=True
    )
    
    return val_loader
    