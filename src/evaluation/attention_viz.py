import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from eval_vit import ViT, ViTBaseConfig
from data.dataset import get_val_dataloader


def attention_rollout(attention_weights):
    """
    Attention Rollout: Average across heads and recursively multiply layers.
    
    Args:
        attention_weights: List of attention tensors from all layers
                          [(B, num_heads, N, N), ...]
    Returns:
        rollout: (B, N, N) attention from output to input space
    """
    # Average across heads for each layer
    attn_layers = [attn.mean(dim=1) for attn in attention_weights]  # Each: (B, N, N)
    
    # Add identity for residual connections
    num_tokens = attn_layers[0].shape[-1]
    eye = torch.eye(num_tokens, device=attn_layers[0].device)
    eye = eye.unsqueeze(0)  # (1, N, N)
    
    # Start with first layer + residual
    result = attn_layers[0] + eye
    
    # Recursively multiply all layers
    for attn in attn_layers[1:]:
        result = torch.matmul(attn + eye, result)
    
    # Normalize
    result = result / result.sum(dim=-1, keepdim=True)
    
    return result


def visualize_attention(model, image, device, save_path=None):
    model.eval()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image_tensor = image.to(device)
    B, C, H, W = image_tensor.shape
    
    # Forward pass (model should store attention_weights)
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get stored attention weights
    attention_weights = model.attention_weights
    
    # Apply attention rollout
    rollout = attention_rollout(attention_weights)  # (B, N, N)
    
    # Get attention from CLS token (index 0) to all patches
    attn_map = rollout[0, 0, 1:].cpu()  # Skip CLS token itself
    
    # Reshape to spatial grid
    num_patches = int(np.sqrt(len(attn_map)))
    attn_map = attn_map.reshape(num_patches, num_patches)
    
    # Resize to image size
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # Prepare image
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(attn_map, cmap='jet')
    axes[1].set_title('Attention Rollout')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    axes[2].imshow(img_np)
    axes[2].imshow(attn_map, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# def main():
#     config = ViTBaseConfig()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model = ViT(config).to(device)
#     checkpoint = torch.load('/home/samitha/Projects/nanoViT/src/log/swa_model_29to_last.pt')
#     model.load_state_dict(checkpoint)
    
#     val_dataloader = get_val_dataloader("/home/samitha/Projects/datasets/imagenet100", batch_size=1)
#     save_dir = Path('experiments/attention_viz/')
#     save_dir.mkdir(parents=True, exist_ok=True)
    
#     val_data = iter(val_dataloader)
#     for i in range(4):
#         image, label = next(val_data)
#         visualize_attention(model, image, device, save_path=save_dir / f'attention_{i}.png')
        

# CPU version
def main():
    config = ViTBaseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = ViT(config)
    model.to(device)
   
    checkpoint = torch.load("src/log/model_300000.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
   
    # for CPU - streaming dataset
    from datasets import load_dataset
    from data.dataset import get_transforms
    
    dataset = load_dataset("clane9/imagenet-100", streaming=True)
    val_transform = get_transforms(is_train=False)
    
    save_dir = Path('experiments/attention_viz/')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate directly without DataLoader
    for i, sample in enumerate(dataset['validation']):
        if i >= 4:
            break
        
        image = val_transform(sample['image']).unsqueeze(0)  # Add batch dim
        label = sample['label']
        
        visualize_attention(model, image, device, save_path=save_dir / f'attention_{i}.png')


if __name__ == '__main__':
    main()