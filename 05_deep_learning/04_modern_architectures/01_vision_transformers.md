# Vision Transformers (ViTs): Attention for Computer Vision

Learn how to implement and use Vision Transformers, the revolutionary architecture that applies the power of transformers to computer vision tasks. Master the cutting-edge approach that's reshaping how we think about image understanding.

## 🎯 What You'll Master

- **Transformer Fundamentals**: Understanding attention mechanisms for vision
- **ViT Architecture**: Complete implementation from scratch
- **Patch Embeddings**: Converting images to sequences
- **Pre-trained Models**: Using state-of-the-art ViT models

## 📚 The Vision Transformer Revolution

### From CNNs to Transformers

**Traditional CNN Approach:**
```
Image → Conv Layers → Pooling → Feature Maps → Classification
```

**Vision Transformer Approach:**
```
Image → Patches → Linear Embeddings → Transformer Encoder → Classification
```

**Key Insight:**
"An image is worth 16x16 words" - treating image patches like tokens in NLP!

Think of ViT like reading a comic book - instead of seeing the whole page at once, you focus on each panel (patch) and understand how they relate to each other through attention!

## 1. Understanding Vision Transformers

### Core Concepts

**Image Patches:**
- Divide image into fixed-size patches (e.g., 16x16 pixels)
- Flatten each patch into a 1D vector
- Treat patches like words in a sentence

**Position Embeddings:**
- Add learnable position information to patches
- Help the model understand spatial relationships
- Critical since transformers don't have inherent spatial awareness

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from typing import Optional, Tuple
import torchvision.transforms as transforms
from PIL import Image

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.proj(attn_output)
        return output, attn_weights

class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Attention with residual connection
        attn_output, attn_weights = self.attention(self.norm1(x))
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x, attn_weights

class VisionTransformer(nn.Module):
    """Complete Vision Transformer implementation"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        embed_dropout=0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token (like [CLS] token in BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(embed_dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Layer norm and classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Extract class token
        logits = self.head(cls_output)
        
        if return_attention:
            return logits, attention_weights
        return logits

# Utility functions for visualization
def visualize_attention(model, image, layer_idx=-1, head_idx=0):
    """Visualize attention maps from ViT"""
    
    model.eval()
    with torch.no_grad():
        # Get attention weights
        logits, attention_weights = model(image.unsqueeze(0), return_attention=True)
        
        # Get attention from specified layer and head
        attn = attention_weights[layer_idx][0, head_idx]  # (seq_len, seq_len)
        
        # Extract attention from class token to patches
        cls_attn = attn[0, 1:]  # Skip class token
        
        # Reshape to spatial dimensions
        img_size = int(math.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(img_size, img_size)
        
        return attn_map.cpu().numpy()

def plot_attention_maps(model, image, num_layers=4, num_heads=4):
    """Plot attention maps from multiple layers and heads"""
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(image.unsqueeze(0), return_attention=True)
    
    fig, axes = plt.subplots(num_layers, num_heads + 1, figsize=(15, 12))
    
    # Original image
    orig_img = image.permute(1, 2, 0)
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    
    for layer_idx in range(num_layers):
        # Show original image in first column
        axes[layer_idx, 0].imshow(orig_img)
        axes[layer_idx, 0].set_title(f'Layer {layer_idx + 1}\nOriginal')
        axes[layer_idx, 0].axis('off')
        
        for head_idx in range(num_heads):
            # Get attention from class token to patches
            attn = attention_weights[layer_idx][0, head_idx]
            cls_attn = attn[0, 1:]  # Skip class token
            
            # Reshape to spatial dimensions
            img_size = int(math.sqrt(len(cls_attn)))
            attn_map = cls_attn.reshape(img_size, img_size)
            
            # Plot attention map
            im = axes[layer_idx, head_idx + 1].imshow(attn_map.cpu().numpy(), cmap='hot')
            axes[layer_idx, head_idx + 1].set_title(f'Head {head_idx + 1}')
            axes[layer_idx, head_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Model variants
def vit_tiny(num_classes=1000):
    """ViT-Tiny model"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes
    )

def vit_small(num_classes=1000):
    """ViT-Small model"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes
    )

def vit_base(num_classes=1000):
    """ViT-Base model"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes
    )

def vit_large(num_classes=1000):
    """ViT-Large model"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes
    )

# Training utilities
class ViTTrainer:
    """Training utilities for Vision Transformers"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, dataloader, optimizer, criterion, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = val_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

# Transfer learning with pre-trained ViT
def load_pretrained_vit(model_name='vit_base_patch16_224', num_classes=10):
    """Load pre-trained ViT and adapt for new task"""
    
    try:
        import timm
        
        # Load pre-trained model
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        print(f"Loaded pre-trained {model_name}")
        
        return model
        
    except ImportError:
        print("timm library not found. Using custom implementation.")
        
        # Use our custom implementation with random weights
        if 'base' in model_name:
            model = vit_base(num_classes)
        elif 'small' in model_name:
            model = vit_small(num_classes)
        elif 'tiny' in model_name:
            model = vit_tiny(num_classes)
        else:
            model = vit_base(num_classes)
        
        print(f"Created {model_name} with random weights")
        return model

# Fine-tuning strategies
def freeze_backbone(model, freeze_layers=8):
    """Freeze early transformer layers for fine-tuning"""
    
    # Freeze patch embedding
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    
    # Freeze position embeddings
    model.pos_embedding.requires_grad = False
    model.cls_token.requires_grad = False
    
    # Freeze specified number of transformer blocks
    for i, block in enumerate(model.transformer_blocks):
        if i < freeze_layers:
            for param in block.parameters():
                param.requires_grad = False
    
    print(f"Frozen first {freeze_layers} transformer layers")

def setup_optimizer_groups(model, lr=1e-4, weight_decay=0.05):
    """Setup optimizer with different learning rates for different parts"""
    
    # Different learning rates for different components
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'head' in n and p.requires_grad],
            'lr': lr * 10,  # Higher learning rate for classifier head
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'head' not in n and p.requires_grad],
            'lr': lr,
            'weight_decay': weight_decay
        }
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer

# Example usage
def demo_vision_transformer():
    """Demonstrate Vision Transformer usage"""
    
    # Create model
    model = vit_base(num_classes=10)
    
    # Create sample input
    batch_size = 2
    sample_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get attention weights
    output_with_attn, attention_weights = model(sample_input, return_attention=True)
    print(f"Number of attention layers: {len(attention_weights)}")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# Advanced ViT variants
class HybridViT(nn.Module):
    """Hybrid ViT with CNN backbone for patch extraction"""
    
    def __init__(self, backbone, img_size=224, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        
        # CNN backbone for feature extraction
        self.backbone = backbone
        
        # Get feature map size from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            feature_map = self.backbone(dummy_input)
            self.feature_size = feature_map.shape[-1]
            self.num_patches = feature_map.shape[-1] ** 2
            backbone_dim = feature_map.shape[1]
        
        # Linear projection to embed_dim
        self.proj = nn.Linear(backbone_dim, embed_dim)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Extract features with CNN backbone
        features = self.backbone(x)  # (batch_size, channels, h, w)
        
        # Flatten spatial dimensions
        batch_size = features.shape[0]
        features = features.flatten(2).transpose(1, 2)  # (batch_size, h*w, channels)
        
        # Project to embedding dimension
        x = self.proj(features)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, _ = block(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

def create_hybrid_vit():
    """Create hybrid ViT with ResNet backbone"""
    
    # Create CNN backbone (ResNet without final layers)
    import torchvision.models as models
    resnet = models.resnet50(pretrained=True)
    
    # Remove final pooling and classification layers
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    
    # Create hybrid ViT
    model = HybridViT(backbone, num_classes=1000)
    
    return model

if __name__ == "__main__":
    # Demo Vision Transformer
    print("=== Vision Transformer Demo ===")
    demo_vision_transformer()
    
    # Test different model sizes
    print("\n=== Model Variants ===")
    models = {
        'ViT-Tiny': vit_tiny(),
        'ViT-Small': vit_small(),
        'ViT-Base': vit_base(),
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total_params:,} parameters")
