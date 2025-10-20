# PyTorch implementation of Hybrid CNN-ViT (ResNet-18 up to layer3 -> Transformer)

import torch
import torch.nn as nn
from torchvision import models  

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-18 up to layer3. For 224x224 input this yields a feature map ~ (B, 256, 14,14).
    """
    def _init_(self, pretrained=False):
        super()._init_()
        resnet = models.resnet18(pretrained=pretrained)
        # keep initial stem
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # keep first 3 blocks (stop before layer4 to maintain higher spatial resolution)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.stem(x)      # -> (B,64,56,56)
        x = self.layer1(x)    # -> (B,64,56,56)
        x = self.layer2(x)    # -> (B,128,28,28)
        x = self.layer3(x)    # -> (B,256,14,14)
        return x

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-ViT:
      - backbone: ResNetFeatureExtractor -> (B, C, Hf, Wf)
      - tokens = flatten(Hf*Wf) -> projection to embed_dim
      - prepend cls token + positional embedding
      - transformer encoder (nn.TransformerEncoder)
      - classification head from cls token
    """
    def _init_(self,
                 img_size=224,
                 backbone_pretrained=False,
                 embed_dim=768,   # model hidden size
                 depth=12,        # transformer layers
                 num_heads=12,    # attention heads
                 mlp_dim=3072,    # feedforward dim in transformer blocks
                 num_classes=100,
                 dropout=0.1):
        super()._init_()
        self.backbone = ResNetFeatureExtractor(pretrained=backbone_pretrained)
        self.backbone_out_channels = 256  # output channels from resnet.layer3
        # compute grid size (for 224 -> 14)
        self.feature_map_size = img_size // 16
        self.num_patches = self.feature_map_size * self.feature_map_size  # tokens from CNN features
        # project channel dimension -> embed_dim
        self.proj = nn.Linear(self.backbone_out_channels, embed_dim)
        # class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=mlp_dim,
                                                   dropout=dropout,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]
        feats = self.backbone(x)                     # (B, C, Hf, Wf)
        B, C, Hf, Wf = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)    # (B, N, C) N = Hf*Wf
        tokens = self.proj(tokens)                   # (B, N, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)# (B,1,embed_dim)
        x = torch.cat((cls_tokens, tokens), dim=1)   # (B, N+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)                        # (N+1, B, embed_dim)
        x = self.transformer(x)                      # (N+1, B, embed_dim)
        x = x.transpose(0, 1)                        # (B, N+1, embed_dim)
        cls_out = x[:, 0]
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)
        return logits

# Example usage / quick smoke test (wrap in if _name_ == '_main_' when running as script)
if __name__ == "_main_":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCNNViT(img_size=224, backbone_pretrained=False, embed_dim=512, depth=6, num_heads=8,
                         mlp_dim=2048, num_classes=100).to(device)
    print("Parameters:", count_params(model))
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print("Output shape:", out.shape)  # (2, num_classes)