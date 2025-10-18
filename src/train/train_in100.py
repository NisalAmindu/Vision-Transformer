import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from dataclasses import dataclass
from PIL import Image
import time
import math
import os
import platform

from vit import ViT


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Custom transform function to ensure RGB (picklable)
def ensure_rgb(img):
    return img.convert("RGB")


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


def get_transforms():
    """Returns train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),  # picklable function
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(ensure_rgb),  # picklable function
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_lr(i, warmup_steps, max_iter, max_lr, min_lr):
    """Learning rate schedule with warmup and cosine decay"""
    # warmup stage : linear
    if i < warmup_steps:
        return (max_lr / warmup_steps) * (i + 1)

    if i > max_iter:
        return min_lr

    # cosine decay
    decay_ratio = (i - warmup_steps) / (max_iter - warmup_steps)
    assert 0 <= decay_ratio <= 1
    c = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + c * (max_lr - min_lr)


@dataclass
class ViTBaseConfig:
    num_classes: int = 100
    img_size: int = 224
    im_channels: int = 3
    patch_size: int = 16

    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 768

    dropout = 0.1

    @property
    def n_patch(self):
        return (self.img_size // self.patch_size) ** 2


def train():
    """Main training function"""
    # Set random seeds
    torch.manual_seed(278)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(278)

    # Hyperparameters
    batch_size = 32
    max_iter = 400000
    warmup_steps = int(max_iter * 0.05)
    max_lr = 3e-4
    min_lr = max_lr * 0.1

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("clane9/imagenet-100")

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    train_ds = ImageNet100(dataset['train'], train_transform)
    val_ds = ImageNet100(dataset['validation'], val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # Create model
    print("Initializing model...")
    vit = ViT(ViTBaseConfig())
    print(f'{sum([p.numel() for p in vit.parameters()]) / 10**6:.2f} M parameters')

    vit = vit.to(device)
    
    # Use torch.compile only on Linux with proper Triton support
    # On Windows, torch.compile has limited support and may fail
    is_windows = platform.system() == 'Windows'
    
    if is_windows or not torch.cuda.is_available():
        print("Using model without torch.compile (Windows or CPU mode)")
        vit_model = vit
    else:
        try:
            print("Attempting to compile model with torch.compile...")
            vit_model = torch.compile(vit)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("Falling back to eager mode...")
            vit_model = vit

    # Create log directory
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")

    with open(log_file, "w") as f:
        pass

    # Setup optimizer
    use_fused = torch.cuda.is_available() and not is_windows
    optimizer = torch.optim.AdamW(
        vit_model.parameters(),
        lr=max_lr,
        weight_decay=0.1,
        fused=use_fused
    )

    # Training loop
    print("Starting training...")
    train_iter = iter(train_loader)

    for step in range(max_iter):
        t0 = time.time()

        # Validation
        if step == 0 or step % 1000 == 0 or step == max_iter:
            val_step = 0
            val_loss = 0.0
            vit_model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = vit_model(x, y)

                    val_loss += loss.detach()
                    val_step += 1
                val_loss /= val_step
            print(f'Validation loss: {val_loss.item():.4f}')
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss.item():.4f}\n")
            vit_model.train()

        # Checkpoints
        if step > 0 and (step % 50000 == 0 or step == max_iter - 1):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': vit.state_dict(),  # Save original model state
                'config': vit.config,
                'step': step,
                'val_loss': val_loss.item(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Training step
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = vit_model(xb, yb)

        loss.backward()

        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(vit_model.parameters(), 1.0)

        # Learning rate schedule
        lr = get_lr(step, warmup_steps, max_iter, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms

        print(f'{step}/{max_iter}  {loss.item():.4f}  {dt:.4f} ms  norm:{norm.item():.4f}  lr:{lr:.4e}')

        with open(log_file, "a") as f:
            f.write(f"{step} train {loss.item():.6f}\n")

    print("Training complete!")


if __name__ == '__main__':
    train()